'''
JA Dual-Critic IPPO: Joint Attention IPPO with separate value heads for
extrinsic (env) and intrinsic (JA) reward streams.

Each reward stream gets its own GAE computation and value loss. Advantages
are normalized independently then summed (RND pattern), so the JA signal
contributes meaningfully regardless of env reward scale.

Only the Python-loop path is implemented (image observations).
'''
import functools
import os
import shutil
from typing import NamedTuple

import hydra
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_ja_dual_image_agent, _get_image_dims
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from agents.ja_utils import jsd_divergence
from common.plot_utils import get_stats, get_metric_names, plot_seed_aggregate
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import batchify, unbatchify, _create_dual_minibatches


class JADualTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value_ext: jnp.ndarray
    value_int: jnp.ndarray
    reward_ext: jnp.ndarray      # env reward only
    reward_int: jnp.ndarray      # beta * (-JSD)
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    ja_reward: jnp.ndarray       # raw -JSD (unscaled), for logging


def _get_obs_type(config):
    return config.get("OBS_TYPE", config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))


def make_train_loop(config, env):
    """Build init and step functions for dual-critic JA-IPPO (Python-loop path)."""
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
    )

    num_envs = config["NUM_ENVS"]
    num_actors = config["NUM_ACTORS"]
    ja_beta_max = config.get("JA_BETA_MAX", 0.01)
    ja_warmup_env_steps = config.get("JA_WARMUP_ENV_STEPS", 200_000)
    env_steps_per_update = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
    ja_warmup_updates = ja_warmup_env_steps / env_steps_per_update
    use_jsd_in_actor = config.get("DUAL_CRITIC_ACTOR_JA", False)
    feed_other_attn = config.get("FEED_OTHER_ATTN", False)

    # Precompute image and feature-map dimensions for attention channel
    img_h, img_w, _ = _get_image_dims(env)
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w,
        stride=config.get("CONV_STRIDE", 2),
        kernel_size=config.get("CONV_KERNEL_SIZE", 3),
        padding=config.get("CONV_PADDING", "SAME"),
        num_blocks=config.get("CONV_NUM_BLOCKS", 4),
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def init_policy(rng):
        rng, init_rng = jax.random.split(rng)
        policy, _ = initialize_ja_dual_image_agent(config, env, init_rng)
        return policy

    def init_state(rng, policy):
        rng, init_rng = jax.random.split(rng)
        _, init_params = initialize_ja_dual_image_agent(config, env, init_rng)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=policy.network.apply, params=init_params, tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        init_hstate = policy.init_hstate(num_actors)
        init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

        if feed_other_attn:
            # Uniform attention: no prior info about where the other agent looked
            init_other_attn = jnp.ones((num_actors, feat_h, feat_w)) / (feat_h * feat_w)
            runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng, init_other_attn)
        else:
            runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng)

        return runner_state

    def make_step_fn(policy):

        def _ppo_update(train_state, traj_batch, adv_ext, adv_int, targets_ext, targets_int, rng):
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, adv_ext, adv_int, targets_ext, targets_int = batch_info

                    def _loss_fn(params, traj_batch, adv_ext, adv_int, targets_ext, targets_int):
                        _, (val_ext, val_int), pi, _, _ = policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch.obs,
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Dual value losses (clipped)
                        def _clipped_value_loss(v_pred, v_old, targets):
                            v_clipped = v_old + (v_pred - v_old).clip(
                                -config["CLIP_EPS"], config["CLIP_EPS"])
                            losses = jnp.square(v_pred - targets)
                            losses_clipped = jnp.square(v_clipped - targets)
                            return jnp.maximum(losses, losses_clipped).mean()

                        value_loss_ext = _clipped_value_loss(val_ext, traj_batch.value_ext, targets_ext)
                        value_loss_int = _clipped_value_loss(val_int, traj_batch.value_int, targets_int)
                        value_loss = value_loss_ext + value_loss_int

                        # Scalarized advantage: normalize independently, then sum (RND pattern)
                        gae = (adv_ext - adv_ext.mean()) / (adv_ext.std() + 1e-8)
                        if use_jsd_in_actor:
                            gae = gae + (adv_int - adv_int.mean()) / (adv_int.std() + 1e-8)

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"])
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss_ext, value_loss_int, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, adv_ext, adv_int, targets_ext, targets_int
                    )
                    grad_norm = jnp.sqrt(
                        sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, grad_norm)

                train_state, init_hstate, traj_batch, adv_ext, adv_int, targets_ext, targets_int, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_dual_minibatches(
                    traj_batch, adv_ext, adv_int, targets_ext, targets_int,
                    init_hstate, num_actors, config["NUM_MINIBATCHES"], perm_rng)

                train_state, minibatch_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, adv_ext, adv_int, targets_ext, targets_int, rng)
                return update_state, minibatch_info

            init_hstate = policy.init_hstate(num_actors)
            update_state = (train_state, init_hstate, traj_batch, adv_ext, adv_int, targets_ext, targets_int, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            return update_state[0], loss_info

        def _augment_obs_with_attn(obs_batch, prev_other_attn):
            """Append upsampled other-agent attention as 4th image channel."""
            upsampled = jax.image.resize(
                prev_other_attn, (num_actors, img_h, img_w), method='nearest',
            )
            rgb = obs_batch.reshape(num_actors, img_h, img_w, 3)
            augmented = jnp.concatenate([rgb, upsampled[..., None]], axis=-1)
            return augmented.reshape(num_actors, -1)

        def _swap_and_reset_attn(attn_map, done_batch):
            """Swap attention maps between agents, reset to uniform on done."""
            attn = attn_map.squeeze(0)  # (num_actors, feat_h, feat_w)
            attn_0 = attn[:num_envs]
            attn_1 = attn[num_envs:]
            # Each agent gets the other's attention
            swapped = jnp.concatenate([attn_1, attn_0], axis=0)
            uniform = jnp.ones((feat_h, feat_w)) / (feat_h * feat_w)
            return jnp.where(done_batch[:, None, None], uniform[None], swapped)

        def _single_step(runner_state, update_steps):
            ja_beta = jnp.minimum(
                ja_beta_max,
                ja_beta_max * update_steps / jnp.maximum(ja_warmup_updates, 1.0),
            )

            def _env_step(runner_state, unused):
                if feed_other_attn:
                    (train_state, env_state, last_obs, last_done, hstate, rng, prev_other_attn) = runner_state
                else:
                    (train_state, env_state, last_obs, last_done, hstate, rng) = runner_state

                rng, act_rng = jax.random.split(rng)

                last_obs_batch = batchify(last_obs, env.agents, num_actors)
                last_done_batch = batchify(last_done, env.agents, num_actors)

                # Augment obs with other agent's previous attention as 4th channel
                if feed_other_attn:
                    last_obs_batch = _augment_obs_with_attn(last_obs_batch, prev_other_attn)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_batch = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_actors).astype(jnp.float32))

                action, (val_ext, val_int), pi, new_hstate, attn_map = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=last_obs_batch.reshape(1, num_actors, -1),
                    done=last_done_batch.reshape(1, num_actors),
                    avail_actions=avail_actions_batch.reshape(1, num_actors, -1),
                    hstate=hstate,
                    rng=act_rng,
                )

                log_prob = pi.log_prob(action)
                action = action.squeeze()
                log_prob = log_prob.squeeze()
                val_ext = val_ext.squeeze()
                val_int = val_int.squeeze()

                env_act = unbatchify(action, env.agents, num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)

                attn_0 = attn_map[:, :num_envs, ...]
                attn_1 = attn_map[:, num_envs:, ...]
                r_ja = -jsd_divergence(attn_0.squeeze(0), attn_1.squeeze(0))
                r_ja = jax.lax.stop_gradient(r_ja)

                reward_batch = batchify(reward, env.agents, num_actors).squeeze()
                r_ja_batch = jnp.concatenate([r_ja, r_ja])

                transition = JADualTransition(
                    done=batchify(new_done, env.agents, num_actors).squeeze(),
                    action=action,
                    value_ext=val_ext,
                    value_int=val_int,
                    reward_ext=reward_batch,
                    reward_int=ja_beta * r_ja_batch,
                    log_prob=log_prob,
                    obs=last_obs_batch,
                    info=info,
                    avail_actions=avail_actions_batch,
                    ja_reward=r_ja_batch,
                )

                if feed_other_attn:
                    new_done_batch = batchify(new_done, env.agents, num_actors).squeeze()
                    new_other_attn = _swap_and_reset_attn(attn_map, new_done_batch)
                    runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng, new_other_attn)
                else:
                    runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            if feed_other_attn:
                (train_state, env_state, last_obs, last_done, hstate, rng, prev_other_attn) = runner_state
            else:
                (train_state, env_state, last_obs, last_done, hstate, rng) = runner_state

            # Bootstrap values
            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_done_batch = batchify(last_done, env.agents, num_actors)
            if feed_other_attn:
                last_obs_batch = _augment_obs_with_attn(last_obs_batch, prev_other_attn)
            last_avail = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail, env.agents, num_actors).astype(jnp.float32))
            _, (last_val_ext, last_val_int), _, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs_batch.reshape(1, num_actors, -1),
                done=last_done_batch.reshape(1, num_actors),
                avail_actions=last_avail_batch.reshape(1, num_actors, -1),
                hstate=hstate,
                rng=jax.random.PRNGKey(0),
            )
            last_val_ext = last_val_ext.squeeze()
            last_val_int = last_val_int.squeeze()

            # Dual GAE
            def _calculate_gae(rewards, values, last_val, dones):
                def _get_advantages(carry, transition_data):
                    gae, next_value = carry
                    done, value, reward = transition_data
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    (dones, values, rewards),
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + values

            adv_ext, targets_ext = _calculate_gae(
                traj_batch.reward_ext, traj_batch.value_ext, last_val_ext, traj_batch.done)
            adv_int, targets_int = _calculate_gae(
                traj_batch.reward_int, traj_batch.value_int, last_val_int, traj_batch.done)

            rng, ppo_rng = jax.random.split(rng)
            train_state, loss_info = _ppo_update(
                train_state, traj_batch, adv_ext, adv_int, targets_ext, targets_int, ppo_rng)

            (total_loss, (value_loss_ext, value_loss_int, policy_loss, entropy)), grad_norm = loss_info

            ja_rew_0 = traj_batch.ja_reward[:, :num_envs]
            jsd_values = -ja_rew_0

            metric = traj_batch.info
            metric["update_steps"] = update_steps
            metric["ja_beta"] = ja_beta
            metric["jsd_mean"] = jsd_values.mean()
            metric["ja_reward_mean"] = ja_rew_0.mean()
            metric["loss_total"] = total_loss[0].mean()
            metric["loss_value_ext"] = value_loss_ext[0].mean()
            metric["loss_value_int"] = value_loss_int[0].mean()
            metric["loss_policy"] = policy_loss[0].mean()
            metric["entropy"] = entropy.mean()
            metric["grad_norm"] = grad_norm.mean()
            metric["raw_env_reward_mean"] = traj_batch.reward_ext[:, :num_envs].mean()
            metric["intrinsic_mean"] = traj_batch.reward_int[:, :num_envs].mean()
            metric["value_ext_mean"] = traj_batch.value_ext.mean()
            metric["value_int_mean"] = traj_batch.value_int.mean()

            if feed_other_attn:
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, prev_other_attn)
            else:
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, update_steps + 1, metric

        @functools.partial(jax.jit, donate_argnums=(0,))
        def step_fn(runner_state, update_steps):
            return _single_step(runner_state, update_steps)

        return step_fn, _single_step

    return make_step_fn, init_policy, init_state


def run_ja_dual_ippo(config, logger):
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    num_seeds = algorithm_config["NUM_SEEDS"]
    num_updates = int(algorithm_config["TOTAL_TIMESTEPS"] // algorithm_config["ROLLOUT_LENGTH"] // algorithm_config["NUM_ENVS"])

    obs_type = _get_obs_type(algorithm_config)

    print(f"[ja_dual_ippo] NUM_UPDATES={num_updates}, NUM_SEEDS={num_seeds}, "
          f"NUM_ENVS={algorithm_config['NUM_ENVS']}, obs_type={obs_type}")

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, num_seeds)

    make_step_fn, init_policy_fn, init_state_fn = make_train_loop(algorithm_config, env)

    num_ckpts = algorithm_config.get("NUM_CHECKPOINTS", 5)
    ckpt_interval = num_updates // max(1, num_ckpts - 1)

    print(f"[ja_dual_ippo] Initializing policy and {num_seeds} seeds...")
    policy = init_policy_fn(rngs[0])
    step_fn, _ = make_step_fn(policy)

    all_seed_metrics = []
    all_seed_ckpts = []
    all_seed_final_params = []

    for seed_idx in range(num_seeds):
        runner_state = init_state_fn(rngs[seed_idx], policy)
        update_steps = jnp.zeros((), dtype=jnp.int32)

        seed_metrics = []
        seed_ckpts = []
        steps_done = 0
        next_ckpt = 0

        print(f"[ja_dual_ippo] Seed {seed_idx}/{num_seeds}: training {num_updates} steps...")
        for ci in range(num_updates):
            runner_state, update_steps, metric = step_fn(runner_state, update_steps)
            seed_metrics.append(metric)
            steps_done += 1

            while next_ckpt <= steps_done and len(seed_ckpts) < num_ckpts:
                seed_ckpts.append(jax.tree.map(jnp.copy, runner_state[0].params))
                next_ckpt += ckpt_interval

            if ci == 0 or ci == num_updates - 1 or (ci + 1) % max(1, num_updates // 10) == 0:
                print(f"[ja_dual_ippo]   step {steps_done}/{num_updates}")

        all_seed_final_params.append(runner_state[0].params)
        all_seed_metrics.append(jax.tree.map(lambda *xs: jnp.stack(xs), *seed_metrics))
        all_seed_ckpts.append(jax.tree.map(lambda *xs: jnp.stack(xs), *seed_ckpts))

    stacked_params = jax.tree.map(lambda *xs: jnp.stack(xs), *all_seed_final_params)
    stacked_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *all_seed_metrics)
    stacked_ckpts = jax.tree.map(lambda *xs: jnp.stack(xs), *all_seed_ckpts)

    print("[ja_dual_ippo] Training complete.")
    out = {
        "final_params": stacked_params,
        "metrics": stacked_metrics,
        "checkpoints": stacked_ckpts,
        "final_ckpt_idx": num_ckpts,
    }

    log_metrics(config, out, logger)
    log_greedy_eval(algorithm_config, env, out, logger, policy)
    log_eval_video(algorithm_config, env, out, logger, policy)
    return out


def log_greedy_eval(algorithm_config, env, out, logger, policy, num_episodes=64):
    """Run greedy and stochastic eval episodes, print per-episode and summary stats."""
    from agents.ja_utils import jsd_divergence, augment_obs_for_eval
    inner_env = env._env
    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    num_seeds = jax.tree.leaves(out["final_params"])[0].shape[0]

    feed_attn = algorithm_config.get("FEED_OTHER_ATTN", False)
    if feed_attn:
        _img_h, _img_w, _ = _get_image_dims(env)
        _feat_h, _feat_w = _compute_resnet_output_dims(
            _img_h, _img_w,
            stride=algorithm_config.get("CONV_STRIDE", 2),
            kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
            padding=algorithm_config.get("CONV_PADDING", "SAME"),
            num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
        )

    for mode_name, greedy in [("greedy", True), ("stochastic", False)]:
        all_returns = []
        all_jsds = []
        for seed_idx in range(num_seeds):
            params = jax.tree.map(lambda x: x[seed_idx], out["final_params"])
            seed_returns = []
            seed_jsds = []
            for ep in range(num_episodes):
                rng = jax.random.PRNGKey(2000 + seed_idx * 10000 + ep)
                rng, reset_rng = jax.random.split(rng)

                obs, env_state = inner_env.reset(reset_rng)
                done = {k: jnp.zeros((1,), dtype=bool) for k in inner_env.agents + ["__all__"]}
                hstate_0 = policy.init_hstate(1)
                hstate_1 = policy.init_hstate(1)

                if feed_attn:
                    prev_attn_0 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)
                    prev_attn_1 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)

                total_reward = 0.0
                ep_jsds = []
                step = 0
                while not done["__all__"] and step < max_steps:
                    avail_actions = inner_env.get_avail_actions(env_state)
                    avail_actions = jax.lax.stop_gradient(avail_actions)

                    obs_0 = obs["agent_0"]
                    obs_1 = obs["agent_1"]
                    if feed_attn:
                        obs_0 = augment_obs_for_eval(obs_0, prev_attn_1, _img_h, _img_w)
                        obs_1 = augment_obs_for_eval(obs_1, prev_attn_0, _img_h, _img_w)

                    rng, rng0, rng1, step_rng = jax.random.split(rng, 4)
                    act_0, hstate_0, attn_0 = policy.get_action_and_attention(
                        params=params,
                        obs=obs_0.reshape(1, 1, -1),
                        done=done["agent_0"].reshape(1, 1),
                        avail_actions=avail_actions["agent_0"].astype(jnp.float32),
                        hstate=hstate_0, rng=rng0, greedy=greedy,
                    )
                    act_1, hstate_1, attn_1 = policy.get_action_and_attention(
                        params=params,
                        obs=obs_1.reshape(1, 1, -1),
                        done=done["agent_1"].reshape(1, 1),
                        avail_actions=avail_actions["agent_1"].astype(jnp.float32),
                        hstate=hstate_1, rng=rng1, greedy=greedy,
                    )

                    if feed_attn:
                        prev_attn_0 = attn_0.squeeze()
                        prev_attn_1 = attn_1.squeeze()

                    jsd_val = float(jsd_divergence(
                        attn_0.squeeze(0), attn_1.squeeze(0)).mean())
                    ep_jsds.append(jsd_val)

                    env_act = {"agent_0": act_0.squeeze(), "agent_1": act_1.squeeze()}
                    obs, env_state, reward, done, info = inner_env.step(step_rng, env_state, env_act)
                    total_reward += float(reward["agent_0"])
                    step += 1

                ep_jsd_mean = float(np.mean(ep_jsds)) if ep_jsds else 0.0
                seed_returns.append(total_reward)
                seed_jsds.append(ep_jsd_mean)
                print(f"[eval] {mode_name} seed={seed_idx} ep={ep}: "
                      f"return={total_reward:.1f}  jsd={ep_jsd_mean:.4f}  steps={step}")

            all_returns.extend(seed_returns)
            all_jsds.extend(seed_jsds)
            print(f"[eval] seed {seed_idx} {mode_name} summary: "
                  f"return={np.mean(seed_returns):.1f} ± {np.std(seed_returns):.1f}  "
                  f"jsd={np.mean(seed_jsds):.4f} ± {np.std(seed_jsds):.4f}")

        ret_mean, ret_std = np.mean(all_returns), np.std(all_returns)
        jsd_mean, jsd_std = np.mean(all_jsds), np.std(all_jsds)
        print(f"[eval] {mode_name} overall ({len(all_returns)} eps): "
              f"return={ret_mean:.1f} ± {ret_std:.1f}  "
              f"jsd={jsd_mean:.4f} ± {jsd_std:.4f}")
        logger.log_item(f"Eval/{mode_name}_return_mean", float(ret_mean), commit=False)
        logger.log_item(f"Eval/{mode_name}_return_std", float(ret_std), commit=False)
        logger.log_item(f"Eval/{mode_name}_jsd_mean", float(jsd_mean), commit=False)
        logger.log_item(f"Eval/{mode_name}_jsd_std", float(jsd_std), commit=False)

    logger.log({}, commit=True)


def _render_lbf_eval_frames(inner_env, ep_states):
    """Render LBF eval frames using the Jumanji matplotlib viewer."""
    import matplotlib
    matplotlib.use("Agg")
    from jumanji.environments.routing.lbf.viewer import LevelBasedForagingViewer

    wrapper = inner_env._env if hasattr(inner_env, '_env') else inner_env
    jumanji_env = wrapper.env if hasattr(wrapper, 'env') else wrapper
    grid_size = jumanji_env._generator.grid_size

    viewer = LevelBasedForagingViewer(grid_size=grid_size, render_mode="rgb_array")
    frames = []
    for s in ep_states:
        rgba = viewer.render(s.env_state)
        frames.append(rgba[:, :, :3].copy())
    viewer.close()
    return frames


def log_eval_video(algorithm_config, env, out, logger, policy):
    """Run eval episodes for all seeds, log videos + attention to wandb."""
    from evaluation.vis_episodes import (
        run_episode_with_states, log_attention_to_wandb, make_attention_video,
        compute_attention_stasis, compute_object_coverage,
    )

    env_name = algorithm_config["ENV_NAME"]
    num_seeds = jax.tree.leaves(out["final_params"])[0].shape[0]
    inner_env = env._env
    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))

    feed_attn = algorithm_config.get("FEED_OTHER_ATTN", False)
    feed_attn_dims = None
    if feed_attn:
        ev_img_h, ev_img_w, _ = _get_image_dims(env)
        ev_feat_h, ev_feat_w = _compute_resnet_output_dims(
            ev_img_h, ev_img_w,
            stride=algorithm_config.get("CONV_STRIDE", 2),
            kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
            padding=algorithm_config.get("CONV_PADDING", "SAME"),
            num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
        )
        feed_attn_dims = (ev_img_h, ev_img_w, ev_feat_h, ev_feat_w)

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    for seed_idx in range(num_seeds):
        final_params = jax.tree.map(lambda x: x[seed_idx], out["final_params"])

        ep_states, attn_data, ep_actions, _ = run_episode_with_states(
            jax.random.PRNGKey(42 + seed_idx), inner_env, final_params, policy,
            final_params, policy, max_steps,
            collect_attention=True,
            feed_other_attn_dims=feed_attn_dims,
        )
        print(f"[ja_dual_ippo] Seed {seed_idx}: eval episode {len(ep_states)} frames collected")

        video_dir = f"{savedir}/videos/seed_{seed_idx}"
        os.makedirs(video_dir, exist_ok=True)

        if env_name in ("lbf", "lbf-image", "lbf-reward-shaping"):
            frames = _render_lbf_eval_frames(inner_env, ep_states)
        elif env_name == "card-game":
            from envs.card_game.rendering import render_card_game_eval_frames
            frames = render_card_game_eval_frames(ep_states, scale=32)
        else:
            from evaluation.vis_episodes import render_episode_frames
            frames = render_episode_frames(ep_states, inner_env.agent_view_size, pixels_per_tile=32)

        tag = f"Eval/seed_{seed_idx}"

        if env_name == "card-game":
            from marl.ja_ippo import _log_card_game_attention_grid, _log_card_game_eval_video
            _log_card_game_attention_grid(
                frames, attn_data, ep_actions, tag, video_dir, logger,
            )
            _log_card_game_eval_video(
                inner_env, policy, final_params, max_steps, tag, video_dir, logger,
                feed_attn_dims=feed_attn_dims, num_episodes=30, fps=3,
            )
        else:
            from moviepy import ImageSequenceClip
            video_path = f"{video_dir}/eval_final.mp4"
            clip = ImageSequenceClip(frames, fps=10)
            clip.write_videofile(video_path, fps=10, codec='libx264', audio=False,
                                 bitrate='8000k', preset='slow')
            logger.log_video(f"{tag}/episode_video", video_path, commit=False)

            log_attention_to_wandb(
                attn_data, logger, step=None, tag_prefix=tag, commit=False,
                frames=frames,
            )

            attn_video_base = f"{video_dir}/eval_attention.mp4"
            make_attention_video(frames, attn_data, filename=attn_video_base, fps=10)
            logger.log_video(f"{tag}/attention_agent0", f"{video_dir}/eval_attention_agent0.mp4", commit=False)
            logger.log_video(f"{tag}/attention_agent1", f"{video_dir}/eval_attention_agent1.mp4", commit=False)
            logger.log_video(f"{tag}/attention_combined", f"{video_dir}/eval_attention_combined.mp4", commit=False)

        # Multi-episode attention metrics
        num_eval_episodes = int(algorithm_config.get("NUM_EVAL_EPISODES", 64))
        is_overcooked = env_name in ("overcooked-v1",)

        attn_feat_h = attn_feat_w = None
        if is_overcooked:
            attn_img_h, attn_img_w, _ = _get_image_dims(env)
            attn_feat_h, attn_feat_w = _compute_resnet_output_dims(
                attn_img_h, attn_img_w,
                stride=algorithm_config.get("CONV_STRIDE", 2),
                kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
                padding=algorithm_config.get("CONV_PADDING", "SAME"),
                num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
            )

        stasis_agent0_vals = []
        stasis_agent1_vals = []
        pct_obj_agent0_vals = []
        pct_obj_agent1_vals = []
        category_accum_agent0: dict[str, float] = {}
        category_accum_agent1: dict[str, float] = {}

        def _accumulate_episode(ep_attn_data, ep_ep_states):
            stasis = compute_attention_stasis(ep_attn_data)
            stasis_agent0_vals.append(stasis["agent_0_stasis"])
            stasis_agent1_vals.append(stasis["agent_1_stasis"])

            if is_overcooked and attn_feat_h is not None and attn_feat_w is not None:
                obj = compute_object_coverage(ep_attn_data, ep_ep_states, attn_feat_h, attn_feat_w)
                pct_obj_agent0_vals.append(obj["agent_0_pct_objects"])
                pct_obj_agent1_vals.append(obj["agent_1_pct_objects"])
                for agent_key, accum in [
                    ("agent_0", category_accum_agent0),
                    ("agent_1", category_accum_agent1),
                ]:
                    cat_mass = obj[f"{agent_key}_category_mass"]
                    for cat, val in cat_mass.items():
                        accum[cat] = accum.get(cat, 0.0) + val

        _accumulate_episode(attn_data, ep_states)

        for ep in range(1, num_eval_episodes):
            ep_rng = jax.random.PRNGKey(42 + seed_idx * 10000 + ep)
            ep_states_extra, attn_data_extra, _, _ = run_episode_with_states(
                ep_rng, inner_env, final_params, policy,
                final_params, policy, max_steps,
                collect_attention=True,
                feed_other_attn_dims=feed_attn_dims,
            )
            _accumulate_episode(attn_data_extra, ep_states_extra)

            if (ep + 1) % max(1, num_eval_episodes // 4) == 0:
                print(f"[ja_dual_ippo] Seed {seed_idx}: eval attention episode {ep + 1}/{num_eval_episodes}")

        n_eps = len(stasis_agent0_vals)
        stasis_a0_mean = float(np.nanmean(stasis_agent0_vals))
        stasis_a0_std = float(np.nanstd(stasis_agent0_vals))
        stasis_a1_mean = float(np.nanmean(stasis_agent1_vals))
        stasis_a1_std = float(np.nanstd(stasis_agent1_vals))

        print(f"[ja_dual_ippo] Seed {seed_idx} stasis ({n_eps} eps): "
              f"agent0={stasis_a0_mean:.4f} +/- {stasis_a0_std:.4f}, "
              f"agent1={stasis_a1_mean:.4f} +/- {stasis_a1_std:.4f}")

        if is_overcooked and pct_obj_agent0_vals:
            pct_a0_mean = float(np.nanmean(pct_obj_agent0_vals))
            pct_a0_std = float(np.nanstd(pct_obj_agent0_vals))
            pct_a1_mean = float(np.nanmean(pct_obj_agent1_vals))
            pct_a1_std = float(np.nanstd(pct_obj_agent1_vals))

            print(f"[ja_dual_ippo] Seed {seed_idx} pct_objects ({n_eps} eps): "
                  f"agent0={pct_a0_mean:.4f} +/- {pct_a0_std:.4f}, "
                  f"agent1={pct_a1_mean:.4f} +/- {pct_a1_std:.4f}")

            for agent_label, accum in [("agent_0", category_accum_agent0),
                                        ("agent_1", category_accum_agent1)]:
                sorted_cats = sorted(accum.items(), key=lambda x: -x[1])[:5]
                parts = [f"{k}={v / n_eps:.3f}" for k, v in sorted_cats]
                print(f"[ja_dual_ippo] Seed {seed_idx} {agent_label} top categories: {', '.join(parts)}")


def log_metrics(config, out, logger):
    '''Save train run output, export CSV, and log mean+std to wandb.'''
    import csv

    train_metrics = out["metrics"]
    metric_names = get_metric_names(config["ENV_NAME"])
    train_stats = get_stats(train_metrics, metric_names)

    algorithm_config = dict(config.algorithm)
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    rollout_length = int(algorithm_config["ROLLOUT_LENGTH"])
    num_envs = int(algorithm_config["NUM_ENVS"])

    plot_seed_aggregate(
        train_stats,
        num_rollout_steps=rollout_length,
        num_envs=num_envs,
        savedir=savedir,
        savename="train_curve",
    )
    import wandb as _wandb
    for name in train_stats:
        png_path = os.path.join(savedir, f"train_curve_{name}.png")
        if os.path.exists(png_path):
            logger.log_item(f"Plots/train_curve_{name}", _wandb.Image(png_path), commit=False)

    num_seeds = train_metrics["returned_episode"].shape[0]
    num_updates = train_metrics["returned_episode"].shape[1]

    # Compute cross-seed mean and std from per-seed means
    # train_stats[k] shape: (num_seeds, num_updates, 2) where [:,:,0] = per-seed mean
    episode_stats_mean = {}
    for k, v in train_stats.items():
        v_arr = np.array(v)
        seed_means = v_arr[:, :, 0]
        episode_stats_mean[k] = np.stack([
            seed_means.mean(axis=0),
            seed_means.std(axis=0),
        ], axis=-1)

    scalar_keys = [
        ("ja_beta",              "JA/beta"),
        ("jsd_mean",             "JA/jsd"),
        ("raw_env_reward_mean",  "Reward/env_raw"),
        ("intrinsic_mean",       "Reward/intrinsic"),
        ("loss_total",           "Loss/total"),
        ("loss_value_ext",       "Loss/value_ext"),
        ("loss_value_int",       "Loss/value_int"),
        ("loss_policy",          "Loss/policy"),
        ("entropy",              "Loss/entropy"),
        ("grad_norm",            "Loss/grad_norm"),
        ("value_ext_mean",       "Value/ext_mean"),
        ("value_int_mean",       "Value/int_mean"),
    ]

    scalar_mean = {}
    scalar_std = {}
    for key, _ in scalar_keys:
        if key in train_metrics:
            vals = np.array(train_metrics[key])
            scalar_mean[key] = np.mean(vals, axis=0)
            scalar_std[key] = np.std(vals, axis=0)

    # Export CSV
    csv_header = ["update", "timestep"]
    for name in metric_names:
        csv_header.extend([f"{name}_mean", f"{name}_std"])
    if config.task["ENV_NAME"] == "overcooked-v1" and "base_return" in metric_names:
        csv_header.append("soups_delivered")
    for key, _ in scalar_keys:
        if key in scalar_mean:
            csv_header.extend([f"{key}_mean", f"{key}_std"])

    csv_path = os.path.join(savedir, "train_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for step in range(num_updates):
            row = [step, (step + 1) * rollout_length * num_envs]
            for name in metric_names:
                stat_data = np.array(train_stats[name])
                seed_means = stat_data[:, step, 0]
                row.extend([float(seed_means.mean()), float(seed_means.std())])
            if config.task["ENV_NAME"] == "overcooked-v1" and "base_return" in metric_names:
                base_data = np.array(train_stats["base_return"])
                row.append(float(base_data[:, step, 0].mean()) / 20.0)
            for key, _ in scalar_keys:
                if key in scalar_mean:
                    row.extend([float(scalar_mean[key][step]), float(scalar_std[key][step])])
            writer.writerow(row)

    print(f"[log_metrics] CSV: {csv_path} ({num_updates} updates, {num_seeds} seeds, {len(csv_header)} cols)")
    import wandb as _wandb
    _wandb.save(csv_path, base_path=savedir)

    # Log to wandb
    print_interval = max(1, num_updates // 20)

    for step in range(num_updates):
        for stat_name, stat_data in episode_stats_mean.items():
            logger.log_item(f"Train/{stat_name}_mean", stat_data[step, 0], train_step=step, commit=False)
            if num_seeds > 1:
                logger.log_item(f"Train/{stat_name}_std", stat_data[step, 1], train_step=step, commit=False)
        if "base_return" in episode_stats_mean and config.task["ENV_NAME"] == "overcooked-v1":
            soups = episode_stats_mean["base_return"][step, 0] / 20.0
            logger.log_item("Train/soups_delivered", soups, train_step=step, commit=False)

        for key, wandb_name in scalar_keys:
            if key in scalar_mean:
                logger.log_item(f"{wandb_name}/mean", float(scalar_mean[key][step]),
                                train_step=step, commit=False)
                if num_seeds > 1:
                    logger.log_item(f"{wandb_name}/std", float(scalar_std[key][step]),
                                    train_step=step, commit=False)

        # Per-seed curves for cross-seed analysis
        for stat_name in train_stats:
            stat_data = np.array(train_stats[stat_name])
            for seed_idx in range(num_seeds):
                logger.log_item(f"Seeds/{stat_name}/seed_{seed_idx}",
                                float(stat_data[seed_idx, step, 0]),
                                train_step=step, commit=False)

        logger.log({}, step=step, commit=True)

        if step % print_interval == 0 or step == num_updates - 1:
            env_steps = (step + 1) * rollout_length * num_envs
            pct = (step + 1) / num_updates * 100
            ret_str = "  ".join(f"{sn}={sd[step, 0]:.2f}" for sn, sd in episode_stats_mean.items())
            jsd = float(scalar_mean.get("jsd_mean", np.zeros(num_updates))[step])
            beta = float(scalar_mean.get("ja_beta", np.zeros(num_updates))[step])
            loss = float(scalar_mean.get("loss_total", np.zeros(num_updates))[step])
            grad = float(scalar_mean.get("grad_norm", np.zeros(num_updates))[step])
            extra = ""
            if "base_return" in episode_stats_mean and config.task["ENV_NAME"] == "overcooked-v1":
                soups = episode_stats_mean["base_return"][step, 0] / 20.0
                extra = f"  soups={soups:.1f}"
            print(f"[{pct:5.1f}%] step={step}/{num_updates}  env_steps={env_steps}  "
                  f"{ret_str}{extra}  jsd={jsd:.4f}  beta={beta:.4f}  loss={loss:.4f}  grad={grad:.3f}")

    logger.commit()

    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
