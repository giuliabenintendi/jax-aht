'''
JA-IPPO: Joint Attention IPPO with shared parameters (Lee et al. 2021).

Both agents share a single network and optimizer (parameter sharing). Cross-agent
coordination comes from:

  1. Cross-agent state routing: each agent receives its partner's actor LSTM
     hidden state h via batchified half-swap.
  2. JA intrinsic reward: r_JA = -JSD(attn_agent_0, attn_agent_1), scaled by
     a beta that ramps linearly from 0 to JA_BETA_MAX over JA_WARMUP_ENV_STEPS.
  3. JATransition stores partner_hstate per timestep so it can be replayed
     during PPO loss recomputation.
'''
import shutil
from typing import NamedTuple

import hydra
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent
from agents.ja_utils import jsd_divergence
from common.plot_utils import get_stats, get_metric_names
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
    partner_hstate: jnp.ndarray  # (NUM_ACTORS, lstm_dim) — stored per timestep
    ja_reward: jnp.ndarray       # (NUM_ACTORS,) — raw JA intrinsic reward (unscaled)


def _construct_partner_hstate(hstate, num_envs, lstm_dim):
    """Construct partner_hstate by swapping agent halves.

    The JA cross-agent query needs each actor's partner hidden state at the
    same env index. Because batchify concatenates agents contiguously
    ([a0_e0..a0_eN, a1_e0..a1_eN]), swapping the two halves aligns each
    position with its partner: position i in the first half (agent 0, env i)
    maps to position i in the second half (agent 1, env i), and vice versa.

    Args:
        hstate: (1, NUM_ACTORS, 4*lstm_dim) packed LSTM states
        num_envs: number of environments
        lstm_dim: LSTM hidden dimension

    Returns:
        partner_h: (1, NUM_ACTORS, lstm_dim) partner's actor h
    """
    actor_h = hstate[..., :lstm_dim]  # (1, NUM_ACTORS, lstm_dim)
    partner_h = jnp.concatenate([actor_h[:, num_envs:, :], actor_h[:, :num_envs, :]], axis=1)
    return partner_h


def make_train(config, env):
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
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

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    obs_type = config.get("OBS_TYPE", config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))
    init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent

    def train(rng):
        # INIT SINGLE SHARED NETWORK
        rng, init_rng = jax.random.split(rng)
        policy, init_params = init_fn(config, env, init_rng)
        lstm_dim = policy.lstm_hidden_dim

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

        # PPO update (single, batchified over both agents)
        def _ppo_update(train_state, traj_batch, advantages, targets, rng):
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # Rerun network with stored partner_hstate
                        _, value, pi, _, _ = policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch.obs,
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0),
                            partner_hstate=traj_batch.partner_hstate,
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Policy gradient loss (clipped)
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

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            # Beta curriculum: linear ramp from 0 to ja_beta_max
            ja_beta = jnp.minimum(
                ja_beta_max,
                ja_beta_max * update_steps / jnp.maximum(ja_warmup_updates, 1.0),
            )

            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, last_done, hstate, rng) = runner_state

                rng, act_rng = jax.random.split(rng)

                # Batchify observations (like IPPO)
                last_obs_batch = batchify(last_obs, env.agents, num_actors)
                last_done_batch = batchify(last_done, env.agents, num_actors)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions_batch = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_actors).astype(jnp.float32))

                # Cross-agent state routing: swap agent halves
                partner_h = _construct_partner_hstate(hstate, num_envs, lstm_dim)

                # Single forward pass for both agents
                action, value, pi, new_hstate, attn_map = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=last_obs_batch.reshape(1, num_actors, -1),
                    done=last_done_batch.reshape(1, num_actors),
                    avail_actions=avail_actions_batch.reshape(1, num_actors, -1),
                    hstate=hstate,
                    rng=act_rng,
                    partner_hstate=partner_h,
                )

                log_prob = pi.log_prob(action)
                action = action.squeeze()
                log_prob = log_prob.squeeze()
                value = value.squeeze()

                # Unbatchify actions for env step
                env_act = unbatchify(action, env.agents, num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)

                # JA intrinsic reward: -JSD between agent_0 and agent_1 attention maps
                # attn_map: (1, NUM_ACTORS, H, W) -> split into agent halves
                attn_0 = attn_map[:, :num_envs, ...]   # (1, num_envs, H, W)
                attn_1 = attn_map[:, num_envs:, ...]    # (1, num_envs, H, W)
                r_ja = -jsd_divergence(attn_0.squeeze(0), attn_1.squeeze(0))  # (num_envs,)
                r_ja = jax.lax.stop_gradient(r_ja)

                # Batchify rewards and add JA intrinsic reward
                reward_batch = batchify(reward, env.agents, num_actors).squeeze()
                # r_ja is per-env, same for both agents -> tile to NUM_ACTORS
                r_ja_batch = jnp.concatenate([r_ja, r_ja])  # (NUM_ACTORS,)
                reward_with_ja = reward_batch + ja_beta * r_ja_batch

                transition = JATransition(
                    done=batchify(new_done, env.agents, num_actors).squeeze(),
                    action=action,
                    value=value,
                    reward=reward_with_ja,
                    log_prob=log_prob,
                    obs=last_obs_batch,
                    info=info,
                    avail_actions=avail_actions_batch,
                    partner_hstate=partner_h.squeeze(0),  # (NUM_ACTORS, lstm_dim)
                    ja_reward=r_ja_batch,
                )
                runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                jax.checkpoint(_env_step), runner_state, None, config["ROLLOUT_LENGTH"]
            )

            # Final value estimate
            (train_state, env_state, last_obs, last_done, hstate, rng) = runner_state

            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_done_batch = batchify(last_done, env.agents, num_actors)
            last_avail = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail, env.agents, num_actors).astype(jnp.float32))
            last_partner_h = _construct_partner_hstate(hstate, num_envs, lstm_dim)

            _, last_val, _, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs_batch.reshape(1, num_actors, -1),
                done=last_done_batch.reshape(1, num_actors),
                avail_actions=last_avail_batch.reshape(1, num_actors, -1),
                hstate=hstate,
                rng=jax.random.PRNGKey(0),
                partner_hstate=last_partner_h,
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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # Single PPO update (batchified over both agents)
            rng, ppo_rng = jax.random.split(rng)
            train_state, loss_info = _ppo_update(
                train_state, traj_batch, advantages, targets, ppo_rng)

            # Loss metrics
            (total_loss, (value_loss, policy_loss, entropy)), grad_norm = loss_info

            # JA diagnostics from the rollout
            # ja_reward: (ROLLOUT_LENGTH, NUM_ACTORS) — first half is agent_0
            ja_rew_0 = traj_batch.ja_reward[:, :num_envs]  # (ROLLOUT_LENGTH, num_envs)
            jsd_values = -ja_rew_0  # JSD is non-negative

            # Build metric dict
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
            # Env reward (agent_0 half, without JA component)
            metric["env_reward_0_mean"] = (
                traj_batch.reward[:, :num_envs] - ja_beta * ja_rew_0
            ).mean()
            metric["value_mean"] = traj_batch.value.mean()

            update_steps += 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

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
            _, update_steps = update_runner_state
            to_store = jnp.logical_or(
                jnp.equal(jnp.mod(update_steps - 1, ckpt_and_eval_interval), 0),
                jnp.equal(update_steps, config["NUM_UPDATES"]),
            )

            def store_ckpt_fn(args):
                _checkpoint_array, _ckpt_idx = args
                new_checkpoint_array = jax.tree.map(
                    lambda c_arr, p: c_arr.at[_ckpt_idx].set(p),
                    _checkpoint_array,
                    update_runner_state[0][0].params,
                )
                return new_checkpoint_array, _ckpt_idx + 1

            def skip_ckpt_fn(args):
                return args

            checkpoint_array, ckpt_idx = jax.lax.cond(
                to_store, store_ckpt_fn, skip_ckpt_fn, (checkpoint_array, ckpt_idx),
            )

            runner_state = (update_runner_state, checkpoint_array, ckpt_idx)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        update_steps = 0
        init_hstate = policy.init_hstate(num_actors)
        init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
        runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng)
        update_runner_state = (runner_state, update_steps)
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


def run_ja_ippo(config, logger):
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    with jax.disable_jit(False):
        num_updates = int(algorithm_config["TOTAL_TIMESTEPS"] // algorithm_config["ROLLOUT_LENGTH"] // algorithm_config["NUM_ENVS"])
        print(f"[ja_ippo] Compiling train fn (NUM_UPDATES={num_updates}, "
              f"NUM_SEEDS={algorithm_config['NUM_SEEDS']}, NUM_ENVS={algorithm_config['NUM_ENVS']})...")
        train_jit = jax.jit(jax.vmap(make_train(algorithm_config, env)))
        print("[ja_ippo] Calling compiled fn (first call triggers XLA compilation)...")
        out = train_jit(rngs)
        print("[ja_ippo] Training complete.")

    log_metrics(config, out, logger)
    log_eval_video(algorithm_config, env, out, logger)
    return out


def log_eval_video(algorithm_config, env, out, logger):
    """Run one eval episode with final params (seed 0), log video + attention to wandb."""
    import os
    from envs.overcooked.adhoc_overcooked_visualizer import AdHocOvercookedVisualizer
    from evaluation.vis_episodes import run_episode_with_states, log_attention_to_wandb

    obs_type = algorithm_config.get("OBS_TYPE",
        algorithm_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))
    init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent

    # Reconstruct policy (same for both agents — shared params)
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(algorithm_config, env, rng)

    # Extract final params from seed 0
    final_params = jax.tree.map(lambda x: x[0], out["final_params"])

    # Unwrap LogWrapper so run_episode_with_states collects WrappedEnvState
    inner_env = env._env

    # Run one eval episode collecting states + attention maps
    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    ep_states, attn_data = run_episode_with_states(
        jax.random.PRNGKey(42), inner_env, final_params, policy,
        final_params, policy, max_steps,
        collect_attention=True,
    )
    print(f"[ja_ippo] Eval episode: {len(ep_states)} frames collected")

    # Render video from collected states
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    video_dir = f"{savedir}/videos"
    os.makedirs(video_dir, exist_ok=True)
    video_path = f"{video_dir}/eval_final.mp4"

    viz = AdHocOvercookedVisualizer()
    viz.animate_mp4(
        [s.env_state for s in ep_states], inner_env.agent_view_size,
        filename=video_path, pixels_per_tile=32, fps=10,
    )
    logger.log_video("Eval/episode_video", video_path)

    # Log attention heatmaps (first / middle / last frame)
    num_updates = out["metrics"]["returned_episode"].shape[1]
    log_attention_to_wandb(attn_data, logger, step=num_updates - 1, tag_prefix="Eval")


def log_metrics(config, out, logger):
    '''Save train run output and log all metrics to wandb.'''
    train_metrics = out["metrics"]
    metric_names = get_metric_names(config["ENV_NAME"])
    train_stats = get_stats(train_metrics, metric_names)

    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    # Scalar metrics to log
    scalar_keys = [
        ("ja_beta", "JA"),
        ("ja_reward_mean", "JA"),
        ("jsd_mean", "JA"),
        ("env_reward_0_mean", "Rewards"),
        ("loss_total", "Losses"),
        ("loss_value", "Losses"),
        ("loss_policy", "Losses"),
        ("entropy", "Losses"),
        ("grad_norm", "Losses"),
        ("value_mean", "Values"),
    ]

    scalar_data = {}
    for key, _ in scalar_keys:
        if key in train_metrics:
            scalar_data[key] = np.mean(np.array(train_metrics[key]), axis=0)

    num_updates = train_metrics["returned_episode"].shape[1]
    print_interval = max(1, num_updates // 20)

    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            logger.log_item(f"Train/{stat_name}", stat_data[step, 0], train_step=step, commit=False)

        for key, prefix in scalar_keys:
            if key in scalar_data:
                logger.log_item(f"{prefix}/{key}", float(scalar_data[key][step]),
                                train_step=step, commit=False)

        logger.log({}, step=step, commit=True)

        if step % print_interval == 0 or step == num_updates - 1:
            env_steps = (step + 1) * int(config.algorithm["ROLLOUT_LENGTH"]) * int(config.algorithm["NUM_ENVS"])
            pct = (step + 1) / num_updates * 100
            ret_str = "  ".join(f"{sn}={sd[step, 0]:.2f}" for sn, sd in train_stats.items())
            jsd = float(scalar_data.get("jsd_mean", np.zeros(num_updates))[step])
            loss = float(scalar_data.get("loss_total", np.zeros(num_updates))[step])
            grad = float(scalar_data.get("grad_norm", np.zeros(num_updates))[step])
            print(f"[{pct:5.1f}%] step={step}/{num_updates}  env_steps={env_steps}  "
                  f"{ret_str}  jsd={jsd:.4f}  loss={loss:.4f}  grad={grad:.3f}")

    logger.commit()

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
