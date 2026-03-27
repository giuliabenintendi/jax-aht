'''
JA-IPPO: Joint Attention IPPO with shared parameters (Lee et al. 2021).

Both agents share a single network and optimizer (parameter sharing). Cross-agent
coordination comes from:

  1. JA intrinsic reward: r_JA = -JSD(attn_agent_0, attn_agent_1), penalty
     for attention divergence, scaled by beta ramping from 0 to JA_BETA_MAX
     over JA_WARMUP_ENV_STEPS. Combined with env reward before normalization.
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

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent, _get_image_dims
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from agents.ja_utils import jsd_divergence
from common.plot_utils import get_stats, get_metric_names, plot_seed_aggregate
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import Transition, batchify, unbatchify, _create_minibatches


class JATransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    ja_reward: jnp.ndarray       # (NUM_ACTORS,) — raw JA intrinsic reward (unscaled)


class RewardNormState(NamedTuple):
    """Running mean/variance for streaming reward normalization (Welford's algorithm)."""
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def reward_norm_init() -> RewardNormState:
    return RewardNormState(
        mean=jnp.zeros(()),
        var=jnp.ones(()),
        count=jnp.zeros(()),
    )


def reward_norm_update(state: RewardNormState, batch: jnp.ndarray) -> RewardNormState:
    """Update running stats with a new batch of rewards."""
    batch_mean = batch.mean()
    batch_var = batch.var()
    batch_count = jnp.array(batch.size, dtype=jnp.float32)

    delta = batch_mean - state.mean
    total_count = state.count + batch_count
    new_mean = state.mean + delta * batch_count / jnp.maximum(total_count, 1.0)
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta ** 2 * state.count * batch_count / jnp.maximum(total_count, 1.0)
    new_var = m2 / jnp.maximum(total_count, 1.0)

    return RewardNormState(mean=new_mean, var=new_var, count=total_count)


def reward_norm_apply(state: RewardNormState, rewards: jnp.ndarray, clip: float = 10.0) -> jnp.ndarray:
    """Normalize rewards using running stats, clip to [-clip, clip]."""
    std = jnp.sqrt(state.var + 1e-8)
    normalized = (rewards - state.mean) / std
    return jnp.clip(normalized, -clip, clip)


def _get_obs_type(config):
    return config.get("OBS_TYPE", config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))


def make_train_scan(config, env):
    """Build a single vmappable train function using lax.scan over updates.

    Used for symbolic observations where memory is not a concern.
    Returns a train(rng) function that can be jit'd and vmapped over seeds.
    """
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
    fixed_partner_pos = -1  # not supported in scan path
    filter_attn_top1 = False
    _fixed_attn = None
    feat_h = feat_w = 0

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    obs_type = _get_obs_type(config)
    agent_init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent

    def train(rng):
        # INIT NETWORK
        rng, init_rng = jax.random.split(rng)
        policy, init_params = agent_init_fn(config, env, init_rng)

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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

        # PPO UPDATE LOGIC
        def _ppo_update(train_state, traj_batch, advantages, targets, rng):
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        _, value, pi, _, _ = policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch.obs,
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
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
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    grad_norm = jnp.sqrt(
                        sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, grad_norm)

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate,
                                                  num_actors, config["NUM_MINIBATCHES"], perm_rng)

                train_state, minibatch_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, minibatch_info

            init_hstate = policy.init_hstate(num_actors)
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            return update_state[0], loss_info

        # SINGLE UPDATE STEP (rollout + PPO)
        def _update_step(update_runner_state, unused):
            (runner_state, update_steps, rew_norm_state, _prev_intrinsic) = update_runner_state

            ja_beta = jnp.minimum(
                ja_beta_max,
                ja_beta_max * update_steps / jnp.maximum(ja_warmup_updates, 1.0),
            )

            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, last_done, hstate, rng) = runner_state

                rng, act_rng = jax.random.split(rng)

                last_obs_batch = batchify(last_obs, env.agents, num_actors)
                last_done_batch = batchify(last_done, env.agents, num_actors)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_batch = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_actors).astype(jnp.float32))

                action, value, pi, new_hstate, attn_map = policy.get_action_value_policy(
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
                value = value.squeeze()

                env_act = unbatchify(action, env.agents, num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)

                # Override agent 1's attention if hardcoded partner
                if fixed_partner_pos >= 0:
                    attn_map = attn_map.at[:, num_envs:, ...].set(
                        jnp.broadcast_to(_fixed_attn[None, None], (attn_map.shape[0], num_envs, feat_h, feat_w)))

                # Filter attention to global argmax
                if filter_attn_top1:
                    flat = attn_map.reshape(*attn_map.shape[:2], -1)
                    top_idx = jnp.argmax(flat, axis=-1, keepdims=True)
                    attn_map = jnp.zeros_like(flat).at[jnp.arange(flat.shape[0])[:, None], jnp.arange(flat.shape[1])[None, :], top_idx].set(1.0).reshape(attn_map.shape)

                attn_0 = attn_map[:, :num_envs, ...]
                attn_1 = attn_map[:, num_envs:, ...]
                r_ja = -jsd_divergence(attn_0.squeeze(0), attn_1.squeeze(0))
                r_ja = jax.lax.stop_gradient(r_ja)

                reward_batch = batchify(reward, env.agents, num_actors).squeeze()
                r_ja_batch = jnp.concatenate([r_ja, r_ja])

                intrinsic = ja_beta * r_ja_batch

                transition = JATransition(
                    done=batchify(new_done, env.agents, num_actors).squeeze(),
                    action=action,
                    value=value,
                    reward=reward_batch,
                    log_prob=log_prob,
                    obs=last_obs_batch,
                    info=info,
                    avail_actions=avail_actions_batch,
                    ja_reward=r_ja_batch,
                )
                runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
                return runner_state, (transition, intrinsic)

            runner_state, (traj_batch, intrinsic_batch) = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            (train_state, env_state, last_obs, last_done, hstate, rng) = runner_state

            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_done_batch = batchify(last_done, env.agents, num_actors)
            last_avail = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail, env.agents, num_actors).astype(jnp.float32))
            _, last_val, _, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs_batch.reshape(1, num_actors, -1),
                done=last_done_batch.reshape(1, num_actors),
                avail_actions=last_avail_batch.reshape(1, num_actors, -1),
                hstate=hstate,
                rng=jax.random.PRNGKey(0),
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            # Save raw env reward before combining
            raw_env_reward = traj_batch.reward

            # Combined reward normalization (matches DeepMind reference)
            combined_raw = raw_env_reward + intrinsic_batch
            rew_norm_state = reward_norm_update(rew_norm_state, combined_raw)
            combined = reward_norm_apply(rew_norm_state, combined_raw)
            traj_batch = traj_batch._replace(reward=combined)

            advantages, targets = _calculate_gae(traj_batch, last_val)

            rng, ppo_rng = jax.random.split(rng)
            train_state, loss_info = _ppo_update(
                train_state, traj_batch, advantages, targets, ppo_rng)

            (total_loss, (value_loss, policy_loss, entropy)), grad_norm = loss_info

            ja_rew_0 = traj_batch.ja_reward[:, :num_envs]
            jsd_values = -ja_rew_0

            metric = traj_batch.info
            metric["update_steps"] = update_steps
            metric["ja_beta"] = ja_beta
            metric["jsd_mean"] = jsd_values.mean()
            metric["ja_reward_mean"] = ja_rew_0.mean()
            metric["loss_total"] = total_loss[0].mean()
            metric["loss_value"] = value_loss[0].mean()
            metric["loss_policy"] = policy_loss[0].mean()
            metric["entropy"] = entropy.mean()
            metric["grad_norm"] = grad_norm.mean()
            metric["raw_env_reward_mean"] = raw_env_reward[:, :num_envs].mean()
            metric["intrinsic_mean"] = intrinsic_batch[:, :num_envs].mean()
            metric["combined_reward_mean"] = combined_raw[:, :num_envs].mean()
            metric["value_mean"] = traj_batch.value.mean()

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            update_steps += 1
            return (runner_state, update_steps, rew_norm_state, intrinsic_batch), metric

        # CHECKPOINT LOGIC (same pattern as ippo.py)
        ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)
        num_ckpts = config["NUM_CHECKPOINTS"]

        def init_ckpt_array(params_pytree):
            return jax.tree.map(
                lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                params_pytree
            )

        def _update_step_with_checkpoint(update_with_ckpt_runner_state, unused):
            (update_runner_state, checkpoint_array, ckpt_idx) = update_with_ckpt_runner_state
            update_runner_state, metric = _update_step(update_runner_state, None)
            _, update_steps, _, _ = update_runner_state

            to_store = jnp.logical_or(
                jnp.equal(jnp.mod(update_steps - 1, ckpt_and_eval_interval), 0),
                jnp.equal(update_steps, config["NUM_UPDATES"]))

            def store_ckpt_fn(args):
                _checkpoint_array, _ckpt_idx = args
                new_checkpoint_array = jax.tree.map(
                    lambda c_arr, p: c_arr.at[_ckpt_idx].set(p),
                    _checkpoint_array,
                    update_runner_state[0][0].params
                )
                return new_checkpoint_array, _ckpt_idx + 1

            def skip_ckpt_fn(args):
                return args

            checkpoint_array, ckpt_idx = jax.lax.cond(
                to_store, store_ckpt_fn, skip_ckpt_fn,
                (checkpoint_array, ckpt_idx),
            )

            runner_state = (update_runner_state, checkpoint_array, ckpt_idx)
            return runner_state, metric

        # RUN TRAINING
        rng, _rng = jax.random.split(rng)
        init_hstate = policy.init_hstate(num_actors)
        init_intrinsic = jnp.zeros((config["ROLLOUT_LENGTH"], num_actors))
        update_runner_state = ((train_state, env_state, obsv, init_done, init_hstate, _rng), 0, reward_norm_init(), init_intrinsic)
        checkpoint_array = init_ckpt_array(train_state.params)
        ckpt_idx = 0
        update_with_ckpt_runner_state = (update_runner_state, checkpoint_array, ckpt_idx)

        runner_state, metrics = jax.lax.scan(
            _update_step_with_checkpoint,
            update_with_ckpt_runner_state,
            xs=None,
            length=config["NUM_UPDATES"],
        )

        update_runner_state, checkpoint_array, final_ckpt_idx = runner_state

        return {
            "final_params": update_runner_state[0][0].params,
            "metrics": metrics,
            "checkpoints": checkpoint_array,
            "final_ckpt_idx": final_ckpt_idx,
        }

    return train


def make_train_loop(config, env):
    """Build init and step functions for JA-IPPO with Python-loop training.

    Used for image observations where memory requires sequential stepping.
    Returns (init_fn, make_step_fn) for the Python loop in run_ja_ippo.
    """
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
    feed_other_attn = config.get("FEED_OTHER_ATTN", False)
    filter_attn_top1 = config.get("FILTER_ATTN_TOP1", False)
    fixed_partner_pos = config.get("ENV_KWARGS", {}).get("fixed_partner_pos", -1)

    # Precompute image and feature-map dimensions for attention channel
    img_h, img_w, _ = _get_image_dims(env)
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w,
        stride=config.get("CONV_STRIDE", 2),
        kernel_size=config.get("CONV_KERNEL_SIZE", 3),
        padding=config.get("CONV_PADDING", "SAME"),
        num_blocks=config.get("CONV_NUM_BLOCKS", 4),
    )

    # Precompute fixed attention map for hardcoded partner
    if fixed_partner_pos >= 0:
        from envs.card_game.rendering import TILE_PIXELS as _TP, GRID_COLS as _GC
        pixel_col = fixed_partner_pos * _TP + _TP // 2
        pixel_row = 1 * _TP + _TP // 2
        fc = min(int(pixel_col / (img_w / feat_w)), feat_w - 1)
        fr = min(int(pixel_row / (img_h / feat_h)), feat_h - 1)
        _fixed_attn = jnp.zeros((feat_h, feat_w))
        _fixed_attn = _fixed_attn.at[fr, fc].set(1.0)
        print(f"[ja_ippo] Fixed partner: pos={fixed_partner_pos}, feature=({fr},{fc}), "
              f"feat_map={feat_h}x{feat_w}, attn_sum={float(_fixed_attn.sum())}")
    else:
        _fixed_attn = None

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    obs_type = _get_obs_type(config)
    agent_init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent

    def init_policy(rng):
        """Create the policy object (called once, not vmapped)."""
        rng, init_rng = jax.random.split(rng)
        policy, _ = agent_init_fn(config, env, init_rng)
        return policy

    def init_state(rng, policy):
        """Initialize runner state for a single seed (vmappable over rng)."""
        rng, init_rng = jax.random.split(rng)
        _, init_params = agent_init_fn(config, env, init_rng)

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
            init_other_attn = jnp.ones((num_actors, feat_h, feat_w)) / (feat_h * feat_w)
            runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng, init_other_attn)
        else:
            runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng)

        return runner_state

    def init(rng):
        """Legacy init: returns (runner_state, policy) for single-seed use."""
        policy = init_policy(rng)
        runner_state = init_state(rng, policy)
        return runner_state, policy

    def make_step_fn(policy):

        def _ppo_update(train_state, traj_batch, advantages, targets, rng):
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        _, value, pi, _, _ = policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch.obs,
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        entropy = pi.entropy().mean()

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    grad_norm = jnp.sqrt(
                        sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, grad_norm)

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate,
                                                  num_actors, config["NUM_MINIBATCHES"], perm_rng)

                train_state, minibatch_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, minibatch_info

            init_hstate = policy.init_hstate(num_actors)
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            return update_state[0], loss_info

        def _augment_obs_with_attn(obs_batch, prev_other_attn):
            """Append upsampled other-agent attention as 4th image channel.

            Preserves any message suffix appended after the image data.
            """
            img_flat_dim = img_h * img_w * 3
            img_part = obs_batch[:, :img_flat_dim]
            extra = obs_batch[:, img_flat_dim:]  # message one-hot or empty

            upsampled = jax.image.resize(
                prev_other_attn, (num_actors, img_h, img_w), method='nearest',
            )
            # Normalize to [0, 1] so attention channel matches RGB scale
            attn_max = jnp.max(upsampled, axis=(-2, -1), keepdims=True)
            upsampled = upsampled / jnp.maximum(attn_max, 1e-8)
            rgb = img_part.reshape(num_actors, img_h, img_w, 3)
            augmented = jnp.concatenate([rgb, upsampled[..., None]], axis=-1)
            result = augmented.reshape(num_actors, -1)
            if extra.shape[1] > 0:
                result = jnp.concatenate([result, extra], axis=-1)
            return result

        def _swap_and_reset_attn(attn_map, done_batch):
            """Swap attention maps between agents, reset to uniform on done."""
            attn = attn_map.squeeze(0)  # (num_actors, feat_h, feat_w)
            attn_0 = attn[:num_envs]
            attn_1 = attn[num_envs:]
            # Each agent gets the other's attention
            swapped = jnp.concatenate([attn_1, attn_0], axis=0)
            uniform = jnp.ones((feat_h, feat_w)) / (feat_h * feat_w)
            return jnp.where(done_batch[:, None, None], uniform[None], swapped)

        def _single_step(runner_state, update_steps, rew_norm_state):
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

                action, value, pi, new_hstate, attn_map = policy.get_action_value_policy(
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
                value = value.squeeze()

                env_act = unbatchify(action, env.agents, num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)

                # Override agent 1's attention if hardcoded partner
                if fixed_partner_pos >= 0:
                    attn_map = attn_map.at[:, num_envs:, ...].set(
                        jnp.broadcast_to(_fixed_attn[None, None], (attn_map.shape[0], num_envs, feat_h, feat_w)))

                # Filter attention to global argmax
                if filter_attn_top1:
                    flat = attn_map.reshape(*attn_map.shape[:2], -1)
                    top_idx = jnp.argmax(flat, axis=-1, keepdims=True)
                    attn_map = jnp.zeros_like(flat).at[jnp.arange(flat.shape[0])[:, None], jnp.arange(flat.shape[1])[None, :], top_idx].set(1.0).reshape(attn_map.shape)

                attn_0 = attn_map[:, :num_envs, ...]
                attn_1 = attn_map[:, num_envs:, ...]
                r_ja = -jsd_divergence(attn_0.squeeze(0), attn_1.squeeze(0))
                r_ja = jax.lax.stop_gradient(r_ja)

                reward_batch = batchify(reward, env.agents, num_actors).squeeze()
                r_ja_batch = jnp.concatenate([r_ja, r_ja])

                intrinsic = ja_beta * r_ja_batch

                transition = JATransition(
                    done=batchify(new_done, env.agents, num_actors).squeeze(),
                    action=action,
                    value=value,
                    reward=reward_batch,
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
                return runner_state, (transition, intrinsic)

            runner_state, (traj_batch, intrinsic_batch) = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            if feed_other_attn:
                (train_state, env_state, last_obs, last_done, hstate, rng, prev_other_attn) = runner_state
            else:
                (train_state, env_state, last_obs, last_done, hstate, rng) = runner_state

            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_done_batch = batchify(last_done, env.agents, num_actors)
            if feed_other_attn:
                last_obs_batch = _augment_obs_with_attn(last_obs_batch, prev_other_attn)
            last_avail = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail, env.agents, num_actors).astype(jnp.float32))
            _, last_val, _, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs_batch.reshape(1, num_actors, -1),
                done=last_done_batch.reshape(1, num_actors),
                avail_actions=last_avail_batch.reshape(1, num_actors, -1),
                hstate=hstate,
                rng=jax.random.PRNGKey(0),
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            # Save raw env reward before combining
            raw_env_reward = traj_batch.reward

            # Combined reward normalization (matches DeepMind reference)
            combined_raw = raw_env_reward + intrinsic_batch
            rew_norm_state = reward_norm_update(rew_norm_state, combined_raw)
            combined = reward_norm_apply(rew_norm_state, combined_raw)
            traj_batch = traj_batch._replace(reward=combined)

            advantages, targets = _calculate_gae(traj_batch, last_val)

            rng, ppo_rng = jax.random.split(rng)
            train_state, loss_info = _ppo_update(
                train_state, traj_batch, advantages, targets, ppo_rng)

            (total_loss, (value_loss, policy_loss, entropy)), grad_norm = loss_info

            ja_rew_0 = traj_batch.ja_reward[:, :num_envs]
            jsd_values = -ja_rew_0

            metric = traj_batch.info
            metric["update_steps"] = update_steps
            metric["ja_beta"] = ja_beta
            metric["jsd_mean"] = jsd_values.mean()
            metric["ja_reward_mean"] = ja_rew_0.mean()
            metric["loss_total"] = total_loss[0].mean()
            metric["loss_value"] = value_loss[0].mean()
            metric["loss_policy"] = policy_loss[0].mean()
            metric["entropy"] = entropy.mean()
            metric["grad_norm"] = grad_norm.mean()
            metric["raw_env_reward_mean"] = raw_env_reward[:, :num_envs].mean()
            metric["intrinsic_mean"] = intrinsic_batch[:, :num_envs].mean()
            metric["combined_reward_mean"] = combined_raw[:, :num_envs].mean()
            metric["value_mean"] = traj_batch.value.mean()

            if feed_other_attn:
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, prev_other_attn)
            else:
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, update_steps + 1, rew_norm_state, metric

        @functools.partial(jax.jit, donate_argnums=(0, 2))
        def step_fn(runner_state, update_steps, rew_norm_state):
            """Single-step wrapper (fallback, same as before)."""
            return _single_step(runner_state, update_steps, rew_norm_state)

        @functools.partial(jax.jit, static_argnums=(3,), donate_argnums=(0, 2))
        def chunked_step_fn(runner_state, update_steps, rew_norm_state, chunk_size):
            """Run chunk_size updates in a single JIT call via lax.scan.

            chunk_size is static — JAX recompiles per distinct value, but there
            are typically only 2-3 distinct sizes (regular + remainder at ckpts).
            """
            def _scan_body(carry, _):
                rs, us, rns = carry
                rs, us, rns, metric = _single_step(rs, us, rns)
                return (rs, us, rns), metric

            (runner_state, update_steps, rew_norm_state), metrics = jax.lax.scan(
                _scan_body,
                (runner_state, update_steps, rew_norm_state),
                None,
                length=chunk_size,
            )
            return runner_state, update_steps, rew_norm_state, metrics

        return step_fn, chunked_step_fn, _single_step

    return init, make_step_fn, init_policy, init_state


def run_ja_ippo(config, logger):
    algorithm_config = dict(config.algorithm)
    # Propagate COMMUNICATION flag into ENV_KWARGS so the env is created with it
    if algorithm_config.get("COMMUNICATION", False):
        env_kwargs = dict(algorithm_config["ENV_KWARGS"])
        env_kwargs["communication"] = True
        algorithm_config["ENV_KWARGS"] = env_kwargs
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    num_seeds = algorithm_config["NUM_SEEDS"]
    num_updates = int(algorithm_config["TOTAL_TIMESTEPS"] // algorithm_config["ROLLOUT_LENGTH"] // algorithm_config["NUM_ENVS"])

    obs_type = _get_obs_type(algorithm_config)

    print(f"[ja_ippo] NUM_UPDATES={num_updates}, NUM_SEEDS={num_seeds}, "
          f"NUM_ENVS={algorithm_config['NUM_ENVS']}, obs_type={obs_type}")

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, num_seeds)

    use_scan = algorithm_config.get("USE_SCAN", False)
    if use_scan:
        train_fn = make_train_scan(algorithm_config, env)
        print(f"[ja_ippo] Using full scan path (compiling {num_seeds} seeds)...")
        train_jit = jax.jit(jax.vmap(train_fn))
        out = train_jit(rngs)
    else:
        init_fn, make_step_fn, init_policy_fn, init_state_fn = make_train_loop(algorithm_config, env)

        num_ckpts = algorithm_config.get("NUM_CHECKPOINTS", 5)
        ckpt_interval = num_updates // max(1, num_ckpts - 1)

        # Compile once for a single seed, then loop over seeds sequentially.
        # This reuses the same compiled step_fn for every seed — no vmap, no
        # per-seed-count recompilation, constant memory regardless of NUM_SEEDS.
        print(f"[ja_ippo] Initializing policy and {num_seeds} seeds...")
        policy = init_policy_fn(rngs[0])
        step_fn, chunked_step_fn, _ = make_step_fn(policy)

        all_seed_metrics = []
        all_seed_ckpts = []
        all_seed_final_params = []

        # Compute chunk boundaries aligned to checkpoint intervals
        chunk_boundaries = []
        for i in range(num_ckpts):
            chunk_boundaries.append(min((i + 1) * ckpt_interval, num_updates))
        if chunk_boundaries[-1] < num_updates:
            chunk_boundaries.append(num_updates)

        for seed_idx in range(num_seeds):
            runner_state = init_state_fn(rngs[seed_idx], policy)
            update_steps = jnp.zeros((), dtype=jnp.int32)
            rew_norm_state = reward_norm_init()

            seed_metrics = []
            seed_ckpts = []
            steps_done = 0

            print(f"[ja_ippo] Seed {seed_idx}/{num_seeds}: training {num_updates} steps...")
            for chunk_end in chunk_boundaries:
                chunk_size = chunk_end - steps_done
                if chunk_size <= 0:
                    continue

                runner_state, update_steps, rew_norm_state, chunk_metrics = chunked_step_fn(
                    runner_state, update_steps, rew_norm_state, chunk_size)
                seed_metrics.append(chunk_metrics)
                steps_done = chunk_end

                # Checkpoint after each chunk
                if len(seed_ckpts) < num_ckpts:
                    seed_ckpts.append(jax.tree.map(jnp.copy, runner_state[0].params))

                print(f"[ja_ippo]   step {steps_done}/{num_updates}")

            all_seed_final_params.append(runner_state[0].params)
            # Concatenate chunk metrics along the update axis (axis 0)
            all_seed_metrics.append(jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *seed_metrics))
            all_seed_ckpts.append(jax.tree.map(lambda *xs: jnp.stack(xs), *seed_ckpts))

        # Stack across seeds: (num_seeds, ...)
        stacked_params = jax.tree.map(lambda *xs: jnp.stack(xs), *all_seed_final_params)
        stacked_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *all_seed_metrics)
        stacked_ckpts = jax.tree.map(lambda *xs: jnp.stack(xs), *all_seed_ckpts)

        print("[ja_ippo] Training complete.")
        out = {
            "final_params": stacked_params,
            "metrics": stacked_metrics,
            "checkpoints": stacked_ckpts,
            "final_ckpt_idx": num_ckpts,
        }

    log_metrics(config, out, logger)
    log_greedy_eval(algorithm_config, env, out, logger)
    log_eval_video(algorithm_config, env, out, logger)
    return out


def log_greedy_eval(algorithm_config, env, out, logger, num_episodes=64):
    """Run greedy and stochastic eval episodes, print per-episode and summary stats."""
    import wandb
    from agents.ja_utils import jsd_divergence, augment_obs_for_eval
    obs_type = _get_obs_type(algorithm_config)
    init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(algorithm_config, env, rng)

    inner_env = env._env
    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    num_seeds = jax.tree.leaves(out["final_params"])[0].shape[0]

    feed_attn = algorithm_config.get("FEED_OTHER_ATTN", False)
    eval_filter_top1 = algorithm_config.get("FILTER_ATTN_TOP1", False)
    if feed_attn:
        _img_h, _img_w, _ = _get_image_dims(env)
        _feat_h, _feat_w = _compute_resnet_output_dims(
            _img_h, _img_w,
            stride=algorithm_config.get("CONV_STRIDE", 2),
            kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
            padding=algorithm_config.get("CONV_PADDING", "SAME"),
            num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
        )

    def _apply_top1(attn):
        """Filter attention to global argmax (single spike)."""
        flat = attn.reshape(-1)
        idx = jnp.argmax(flat)
        return jnp.zeros_like(flat).at[idx].set(1.0).reshape(attn.shape)

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

                    if eval_filter_top1:
                        attn_0 = _apply_top1(attn_0.squeeze())[None, None]
                        attn_1 = _apply_top1(attn_1.squeeze())[None, None]

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

        # Per-episode results table
        table = wandb.Table(
            columns=["episode", "return", "jsd"],
            data=[[i, all_returns[i], all_jsds[i]] for i in range(len(all_returns))],
        )
        logger.log({
            f"Eval/{mode_name}_returns_chart": wandb.plot.bar(
                table, "episode", "return", title=f"{mode_name} per-episode return"),
            f"Eval/{mode_name}_jsd_chart": wandb.plot.bar(
                table, "episode", "jsd", title=f"{mode_name} per-episode JSD"),
        }, commit=False)

    logger.log({}, commit=True)


def _draw_box(cell, row, col, tile_h, tile_w, color, thickness):
    """Draw a thick border around a tile at grid (row, col) on an upscaled frame."""
    y0 = row * tile_h
    x0 = col * tile_w
    cell[y0:y0 + thickness, x0:x0 + tile_w] = color
    cell[y0 + tile_h - thickness:y0 + tile_h, x0:x0 + tile_w] = color
    cell[y0:y0 + tile_h, x0:x0 + thickness] = color
    cell[y0:y0 + tile_h, x0 + tile_w - thickness:x0 + tile_w] = color


def _draw_choice_on_cell(cell, choice_pos, agent_idx, scale, card_row=1, card_col=None):
    """Draw white borders around the agent tile and its chosen card.

    Derives grid dimensions from the frame shape and TILE_PIXELS.
    Works for both static (3x5) and dynamic (6x10) grids.
    """
    tile_h = scale * 7
    tile_w = scale * 7
    frame_h, frame_w = cell.shape[:2]
    grid_rows = frame_h // tile_h
    grid_cols = frame_w // tile_w
    thickness = max(2, scale // 8)
    white = [255, 255, 255]

    col = card_col if card_col is not None else choice_pos
    _draw_box(cell, card_row, col, tile_h, tile_w, white, thickness)
    agent_row = 0 if agent_idx == 0 else grid_rows - 1
    agent_col = grid_cols // 2
    _draw_box(cell, agent_row, agent_col, tile_h, tile_w, white, thickness)


def _draw_message_on_cell(cell, msg_pos, scale, color=None):
    """Draw a border around the card at msg_pos (row 1) in the partner's color."""
    tile_h = scale * 7
    tile_w = scale * 7
    frame_h, frame_w = cell.shape[:2]
    grid_rows = frame_h // tile_h
    grid_cols = frame_w // tile_w
    thickness = max(2, scale // 8)
    if color is None:
        color = [139, 90, 43]  # fallback brown
    _draw_box(cell, 1, msg_pos, tile_h, tile_w, color, thickness)


def _log_card_game_attention_grid(frames, attn_data, ep_actions, tag, video_dir, logger,
                                  ep_messages=None, card_positions=None,
                                  card_permutation=None):
    """Log a 2×T grid image: row 0 = agent 0 attention, row 1 = agent 1 attention.

    Each cell shows the scene with the attention heatmap overlaid.
    The last column shows the agents' card choices as thick colored borders
    drawn on top of the spot marker.
    """
    import os
    import wandb
    import numpy as np
    from evaluation.vis_episodes import _overlay_attention

    maps_0 = attn_data.get("agent_0", [])
    maps_1 = attn_data.get("agent_1", [])
    if not maps_0 or not maps_1:
        print("[card_game] Missing attention maps, skipping grid.")
        return

    n_steps = min(len(maps_0), len(maps_1), len(frames) - 1)

    # frames come from render_card_game_eval_frames(scale=32)
    # tile_px = scale * 7. Recover scale from the frame height.
    # frame_h = grid_rows * 7 * scale → scale = frame_h / (grid_rows * 7)
    # But we can just use: one tile = frame_h / grid_rows pixels, scale = tile / 7
    # Since frames[0] was rendered at scale=32, tile = 32*7=224 px per grid cell
    scale = 32

    agent0_color = np.array([255, 140, 0], dtype=np.uint8)    # orange
    agent1_color = np.array([255, 0, 255], dtype=np.uint8)    # magenta

    # Get choices from the last action taken
    last_action = ep_actions[-1] if ep_actions else (-1, -1)

    # Use initial frame for all overlays (cards don't change within episode;
    # the last frame in ep_states is the auto-reset with new shuffle)
    base_frame = frames[0]

    # Build overlay frames for each agent at each timestep
    row_0 = []  # agent 0 attention (Oranges)
    row_1 = []  # agent 1 attention (RdPu)
    for t in range(n_steps):
        frame = base_frame
        cell_0 = _overlay_attention(frame, maps_0[t], "Oranges", alpha=0.6).copy()
        cell_1 = _overlay_attention(frame, maps_1[t], "RdPu", alpha=0.6).copy()

        # Draw partner's message border in partner's color
        if ep_messages and t < len(ep_messages):
            _draw_message_on_cell(cell_0, ep_messages[t][1], scale, color=agent1_color)
            _draw_message_on_cell(cell_1, ep_messages[t][0], scale, color=agent0_color)

        # Draw choice borders on the last timestep
        if t == n_steps - 1 and last_action[0] >= 0:
            if card_positions is not None:
                r0, c0 = int(card_positions[last_action[0]][0]), int(card_positions[last_action[0]][1])
                _draw_choice_on_cell(cell_0, last_action[0], 0, scale, card_row=r0, card_col=c0)
            elif card_permutation is not None:
                col0 = int(np.where(card_permutation == last_action[0])[0][0])
                _draw_choice_on_cell(cell_0, col0, 0, scale)
            else:
                _draw_choice_on_cell(cell_0, last_action[0], 0, scale)
        if t == n_steps - 1 and last_action[1] >= 0:
            if card_positions is not None:
                r1, c1 = int(card_positions[last_action[1]][0]), int(card_positions[last_action[1]][1])
                _draw_choice_on_cell(cell_1, last_action[1], 1, scale, card_row=r1, card_col=c1)
            elif card_permutation is not None:
                col1 = int(np.where(card_permutation == last_action[1])[0][0])
                _draw_choice_on_cell(cell_1, col1, 1, scale)
            else:
                _draw_choice_on_cell(cell_1, last_action[1], 1, scale)

        row_0.append(cell_0)
        row_1.append(cell_1)

    # Concatenate: each row is T frames side by side, then stack 2 rows
    padding = 4
    pad_color = np.array([255, 255, 255], dtype=np.uint8)
    cell_h, cell_w = row_0[0].shape[:2]

    grid_w = n_steps * cell_w + (n_steps - 1) * padding
    grid_h = 2 * cell_h + padding
    grid = np.full((grid_h, grid_w, 3), pad_color, dtype=np.uint8)

    for t in range(n_steps):
        x = t * (cell_w + padding)
        grid[0:cell_h, x:x + cell_w] = row_0[t]
        grid[cell_h + padding:grid_h, x:x + cell_w] = row_1[t]

    # Save locally and log to wandb
    os.makedirs(video_dir, exist_ok=True)
    grid_path = f"{video_dir}/attention_grid.png"
    from PIL import Image
    Image.fromarray(grid).save(grid_path)
    print(f"[card_game] Saved attention grid: {grid_path} ({grid_w}×{grid_h} px)")

    logger.log({f"{tag}/attention_grid": wandb.Image(grid_path)}, commit=False)


def _log_card_game_eval_video(inner_env, policy, params, max_steps, tag, video_dir, logger,
                               feed_attn_dims=None,
                               fixed_partner_attn=None, filter_top1=False,
                               num_episodes=30, fps=3):
    """Run multiple card game episodes and save a video with attention spots and choices."""
    import os
    import wandb
    import numpy as np
    from PIL import Image
    from moviepy import ImageSequenceClip
    from evaluation.vis_episodes import run_episode_with_states, _overlay_attention
    from envs.card_game.rendering import render_card_game, GRID_ROWS, GRID_COLS, TILE_PIXELS

    scale = 20
    padding = 4
    all_video_frames = []

    for ep in range(num_episodes):
        ep_rng = jax.random.PRNGKey(100 + ep)
        ep_states, attn_data, ep_actions, ep_messages = run_episode_with_states(
            ep_rng, inner_env, params, policy,
            params, policy, max_steps,
            collect_attention=True,
            feed_other_attn_dims=feed_attn_dims,
            fixed_partner_attn=fixed_partner_attn,
        )

        if filter_top1:
            def _top1_np(attn):
                a = np.array(attn).squeeze()
                out = np.zeros_like(a)
                out.flat[np.argmax(a)] = 1.0
                return out
            for agent_key in ("agent_0", "agent_1"):
                attn_data[agent_key] = [_top1_np(m) for m in attn_data.get(agent_key, [])]

        maps_0 = attn_data.get("agent_0", [])
        maps_1 = attn_data.get("agent_1", [])
        if not maps_0 or not maps_1:
            continue

        n_steps = min(len(maps_0), len(maps_1))
        es0 = ep_states[0].env_state
        if hasattr(es0, 'card_permutation'):
            base_img = render_card_game(es0.card_permutation)
        else:
            from envs.card_game.rendering_dynamic import render_card_game as render_dynamic
            base_img = render_dynamic(es0.card_positions, es0.card_present)
        base_np = np.array(base_img)
        base_up = np.array(Image.fromarray(base_np).resize(
            (base_np.shape[1] * scale, base_np.shape[0] * scale), Image.NEAREST))

        last_action = ep_actions[-1] if ep_actions else (-1, -1)

        for t in range(n_steps):
            cell_0 = _overlay_attention(base_up, maps_0[t], "Oranges", alpha=0.6).copy()
            cell_1 = _overlay_attention(base_up, maps_1[t], "RdPu", alpha=0.6).copy()

            # Draw partner's message border in partner's color
            if ep_messages and t < len(ep_messages):
                a0_color = [255, 140, 0]    # orange
                a1_color = [255, 0, 255]    # magenta
                _draw_message_on_cell(cell_0, ep_messages[t][1], scale, color=a1_color)
                _draw_message_on_cell(cell_1, ep_messages[t][0], scale, color=a0_color)

            # Draw choice borders on decision step
            if t == n_steps - 1 and last_action[0] >= 0:
                es_ep = ep_states[0].env_state
                if hasattr(es_ep, 'card_positions'):
                    _cp = np.array(es_ep.card_positions)
                    r0, c0 = int(_cp[last_action[0]][0]), int(_cp[last_action[0]][1])
                    _draw_choice_on_cell(cell_0, last_action[0], 0, scale, card_row=r0, card_col=c0)
                    if last_action[1] >= 0:
                        r1, c1 = int(_cp[last_action[1]][0]), int(_cp[last_action[1]][1])
                        _draw_choice_on_cell(cell_1, last_action[1], 1, scale, card_row=r1, card_col=c1)
                elif hasattr(es_ep, 'card_permutation'):
                    _perm = np.array(es_ep.card_permutation)
                    col0 = int(np.where(_perm == last_action[0])[0][0])
                    _draw_choice_on_cell(cell_0, col0, 0, scale)
                    if last_action[1] >= 0:
                        col1 = int(np.where(_perm == last_action[1])[0][0])
                        _draw_choice_on_cell(cell_1, col1, 1, scale)
                else:
                    _draw_choice_on_cell(cell_0, last_action[0], 0, scale)
                    _draw_choice_on_cell(cell_1, last_action[1], 1, scale)

            # Stack vertically: agent 0 on top, agent 1 on bottom
            cell_h, cell_w = cell_0.shape[:2]
            frame = np.full((2 * cell_h + padding, cell_w, 3), 255, dtype=np.uint8)
            frame[:cell_h] = cell_0
            frame[cell_h + padding:] = cell_1
            all_video_frames.append(frame)

    if not all_video_frames:
        print("[card_game] No frames for eval video")
        return

    os.makedirs(video_dir, exist_ok=True)
    video_path = f"{video_dir}/eval_card_game.mp4"
    clip = ImageSequenceClip(all_video_frames, fps=fps)
    clip.write_videofile(video_path, fps=fps, codec='libx264', audio=False,
                         bitrate='8000k', preset='slow')
    logger.log_video(f"{tag}/eval_video", video_path, commit=False)
    print(f"[card_game] Saved eval video: {video_path} ({len(all_video_frames)} frames, {len(all_video_frames)/fps:.0f}s)")


def _render_lbf_eval_frames(inner_env, ep_states):
    """Render LBF eval frames using the Jumanji matplotlib viewer for quality."""
    import matplotlib
    matplotlib.use("Agg")
    from jumanji.environments.routing.lbf.viewer import LevelBasedForagingViewer

    # Get grid_size from the wrapper or underlying jumanji env
    wrapper = inner_env._env if hasattr(inner_env, '_env') else inner_env
    jumanji_env = wrapper.env if hasattr(wrapper, 'env') else wrapper
    grid_size = jumanji_env._generator.grid_size

    viewer = LevelBasedForagingViewer(grid_size=grid_size, render_mode="rgb_array")
    frames = []
    for s in ep_states:
        rgba = viewer.render(s.env_state)
        # RGBA -> RGB
        frames.append(rgba[:, :, :3].copy())
    viewer.close()
    return frames


def log_eval_video(algorithm_config, env, out, logger):
    """Run eval episodes for all seeds, log videos + attention to wandb."""
    import os
    from evaluation.vis_episodes import (
        run_episode_with_states, log_attention_to_wandb, make_attention_video,
        compute_attention_stasis, compute_object_coverage,
    )

    env_name = algorithm_config["ENV_NAME"]

    obs_type = _get_obs_type(algorithm_config)
    init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent

    # Reconstruct policy (same for both agents — shared params)
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(algorithm_config, env, rng)

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

    # Build fixed partner attention for eval visualization
    fixed_partner_pos_eval = algorithm_config.get("ENV_KWARGS", {}).get("fixed_partner_pos", -1)
    fixed_partner_attn_eval = None
    if fixed_partner_pos_eval >= 0:
        ev_img_h_fp, ev_img_w_fp, _ = _get_image_dims(env)
        ev_fh_fp, ev_fw_fp = _compute_resnet_output_dims(
            ev_img_h_fp, ev_img_w_fp,
            stride=algorithm_config.get("CONV_STRIDE", 2),
            kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
            padding=algorithm_config.get("CONV_PADDING", "SAME"),
            num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
        )
        from envs.card_game.rendering import TILE_PIXELS as _TP_eval
        pixel_col = fixed_partner_pos_eval * _TP_eval + _TP_eval // 2
        pixel_row = 1 * _TP_eval + _TP_eval // 2
        fc = min(int(pixel_col / (ev_img_h_fp / ev_fh_fp)), ev_fw_fp - 1)
        fr = min(int(pixel_row / (ev_img_h_fp / ev_fh_fp)), ev_fh_fp - 1)
        fixed_partner_attn_eval = jnp.zeros((ev_fh_fp, ev_fw_fp))
        fixed_partner_attn_eval = fixed_partner_attn_eval.at[fr, fc].set(1.0)

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    for seed_idx in range(num_seeds):
        final_params = jax.tree.map(lambda x: x[seed_idx], out["final_params"])

        ep_states, attn_data, ep_actions, ep_messages = run_episode_with_states(
            jax.random.PRNGKey(42 + seed_idx), inner_env, final_params, policy,
            final_params, policy, max_steps,
            collect_attention=True,
            feed_other_attn_dims=feed_attn_dims,
            fixed_partner_attn=fixed_partner_attn_eval,
        )
        # Apply top1 filter to collected attention maps (match training behavior)
        if algorithm_config.get("FILTER_ATTN_TOP1", False):
            import numpy as _np
            def _top1_np(attn):
                a = _np.array(attn).squeeze()
                out = _np.zeros_like(a)
                out.flat[_np.argmax(a)] = 1.0
                return out
            for agent_key in ("agent_0", "agent_1"):
                attn_data[agent_key] = [_top1_np(m) for m in attn_data.get(agent_key, [])]

        print(f"[ja_ippo] Seed {seed_idx}: eval episode {len(ep_states)} frames collected")

        video_dir = f"{savedir}/videos/seed_{seed_idx}"
        os.makedirs(video_dir, exist_ok=True)

        # Render frames from episode states
        if env_name in ("lbf", "lbf-image", "lbf-reward-shaping"):
            frames = _render_lbf_eval_frames(inner_env, ep_states)
        elif env_name == "card-game":
            from envs.card_game.rendering import render_card_game_eval_frames
            frames = render_card_game_eval_frames(ep_states, scale=32)
        elif env_name == "card-game-dynamic":
            from envs.card_game.rendering_dynamic import render_card_game_eval_frames as render_dynamic_frames
            frames = render_dynamic_frames(ep_states, scale=32)
        else:
            from evaluation.vis_episodes import render_episode_frames
            frames = render_episode_frames(ep_states, inner_env.agent_view_size, pixels_per_tile=32)

        tag = f"Eval/seed_{seed_idx}"

        if env_name in ("card-game", "card-game-dynamic"):
            # Card game: 2×T grid image + multi-episode video
            # Pass card layout for border drawing
            import numpy as _np
            _card_pos = None
            _card_perm = None
            es0 = ep_states[0].env_state
            if hasattr(es0, 'card_positions'):
                _card_pos = _np.array(es0.card_positions)
            if hasattr(es0, 'card_permutation'):
                _card_perm = _np.array(es0.card_permutation)
            _log_card_game_attention_grid(
                frames, attn_data, ep_actions, tag, video_dir, logger,
                ep_messages=ep_messages, card_positions=_card_pos,
                card_permutation=_card_perm,
            )
            _log_card_game_eval_video(
                inner_env, policy, final_params, max_steps, tag, video_dir, logger,
                feed_attn_dims=feed_attn_dims,
                fixed_partner_attn=fixed_partner_attn_eval,
                filter_top1=algorithm_config.get("FILTER_ATTN_TOP1", False),
                num_episodes=30, fps=3,
            )
        else:
            # Other envs: videos + attention overlays
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
        import numpy as np

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
                print(f"[ja_ippo] Seed {seed_idx}: eval attention episode {ep + 1}/{num_eval_episodes}")

        n_eps = len(stasis_agent0_vals)
        stasis_a0_mean = float(np.nanmean(stasis_agent0_vals))
        stasis_a0_std = float(np.nanstd(stasis_agent0_vals))
        stasis_a1_mean = float(np.nanmean(stasis_agent1_vals))
        stasis_a1_std = float(np.nanstd(stasis_agent1_vals))

        print(f"[ja_ippo] Seed {seed_idx} stasis ({n_eps} eps): "
              f"agent0={stasis_a0_mean:.4f} +/- {stasis_a0_std:.4f}, "
              f"agent1={stasis_a1_mean:.4f} +/- {stasis_a1_std:.4f}")

        logger.log({
            f"{tag}/stasis_agent0_mean": stasis_a0_mean,
            f"{tag}/stasis_agent0_std": stasis_a0_std,
            f"{tag}/stasis_agent1_mean": stasis_a1_mean,
            f"{tag}/stasis_agent1_std": stasis_a1_std,
        }, commit=False)

        if is_overcooked and pct_obj_agent0_vals:
            pct_a0_mean = float(np.nanmean(pct_obj_agent0_vals))
            pct_a0_std = float(np.nanstd(pct_obj_agent0_vals))
            pct_a1_mean = float(np.nanmean(pct_obj_agent1_vals))
            pct_a1_std = float(np.nanstd(pct_obj_agent1_vals))

            print(f"[ja_ippo] Seed {seed_idx} pct_objects ({n_eps} eps): "
                  f"agent0={pct_a0_mean:.4f} +/- {pct_a0_std:.4f}, "
                  f"agent1={pct_a1_mean:.4f} +/- {pct_a1_std:.4f}")

            logger.log({
                f"{tag}/pct_objects_agent0_mean": pct_a0_mean,
                f"{tag}/pct_objects_agent0_std": pct_a0_std,
                f"{tag}/pct_objects_agent1_mean": pct_a1_mean,
                f"{tag}/pct_objects_agent1_std": pct_a1_std,
            }, commit=False)

            for agent_label, accum in [("agent_0", category_accum_agent0),
                                        ("agent_1", category_accum_agent1)]:
                sorted_cats = sorted(accum.items(), key=lambda x: -x[1])[:5]
                parts = [f"{k}={v / n_eps:.3f}" for k, v in sorted_cats]
                print(f"[ja_ippo] Seed {seed_idx} {agent_label} top categories: {', '.join(parts)}")
                for cat, val in sorted_cats:
                    logger.log({f"{tag}/{agent_label}_attn_{cat}": val / n_eps}, commit=False)



def log_metrics(config, out, logger):
    '''Save train run output, export CSV, and log mean±std to wandb.'''
    import csv

    train_metrics = out["metrics"]
    metric_names = get_metric_names(config["ENV_NAME"])
    train_stats = get_stats(train_metrics, metric_names)

    algorithm_config = dict(config.algorithm)
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    rollout_length = int(algorithm_config["ROLLOUT_LENGTH"])
    num_envs = int(algorithm_config["NUM_ENVS"])

    # Save mean±std training curve PNGs
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

    # Scalar metrics: (metric_key, wandb_name)
    # Dropped ja_reward_mean (= -jsd) and intrinsic_mean (= -beta*jsd) as redundant
    scalar_keys = [
        ("ja_beta",              "JA/beta"),
        ("jsd_mean",             "JA/jsd"),
        ("raw_env_reward_mean",  "Reward/env_raw"),
        ("combined_reward_mean", "Reward/combined_raw"),
        ("loss_total",           "Loss/total"),
        ("loss_value",           "Loss/value"),
        ("loss_policy",          "Loss/policy"),
        ("entropy",              "Loss/entropy"),
        ("grad_norm",            "Loss/grad_norm"),
        ("value_mean",           "Value/mean"),
    ]

    scalar_mean = {}
    scalar_std = {}
    for key, _ in scalar_keys:
        if key in train_metrics:
            vals = np.array(train_metrics[key])
            scalar_mean[key] = np.mean(vals, axis=0)
            scalar_std[key] = np.std(vals, axis=0)

    # --- Export CSV ---
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

    # --- Log to wandb ---
    print_interval = max(1, num_updates // 20)

    for step in range(num_updates):
        # Episode metrics
        for stat_name, stat_data in episode_stats_mean.items():
            logger.log_item(f"Train/{stat_name}_mean", stat_data[step, 0], train_step=step, commit=False)
            if num_seeds > 1:
                logger.log_item(f"Train/{stat_name}_std", stat_data[step, 1], train_step=step, commit=False)
        if "base_return" in episode_stats_mean and config.task["ENV_NAME"] == "overcooked-v1":
            soups = episode_stats_mean["base_return"][step, 0] / 20.0
            logger.log_item("Train/soups_delivered", soups, train_step=step, commit=False)

        # Scalar metrics
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
