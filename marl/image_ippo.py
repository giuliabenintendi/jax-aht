'''
Image IPPO: Parameter-shared IPPO with ResNet+LSTM image architecture.

Ablation baseline for JA-IPPO. Same ResNet encoder, same LSTM, same PPO
hyperparameters — but no attention mechanism and no JSD intrinsic reward.
Uses parameter sharing (single network for both agents, like standard IPPO).
'''
import hydra
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_image_agent
from common.train_logging import (
    IMAGE_IPPO_SCALAR_KEYS,
    log_live_chunk_metrics,
    report_basic_training_outputs,
)
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.checkpointing import compute_chunk_boundaries, finalize_best_checkpoints
from marl.ppo_core import (
    calculate_gae,
    compute_last_value,
    configure_training_dims,
    make_optimizer,
    reward_norm_apply,
    reward_norm_init,
    reward_norm_update,
    run_ppo_epochs,
)
from marl.ppo_utils import Transition, batchify, unbatchify


def make_train(config, env):
    """Build init and step functions for Image IPPO training.

    Returns (init_fn, step_fn) where:
      - init_fn(rng) -> (runner_state, policy) sets up network, optimizer, env
      - step_fn(runner_state, update_steps) -> (runner_state, update_steps, metric)
        runs one rollout + PPO update
    """
    configure_training_dims(config, env)

    num_envs = config["NUM_ENVS"]
    num_actors = config["NUM_ACTORS"]
    normalize_rewards = bool(config.get("NORMALIZE_REWARDS", False))
    rew_shaping_horizon = float(config.get("REW_SHAPING_HORIZON", 0.0))
    env_steps_per_update = int(config["ROLLOUT_LENGTH"]) * int(config["NUM_ENVS"])

    def init(rng):
        rng, init_rng = jax.random.split(rng)
        policy, init_params = initialize_image_agent(config, env, init_rng)

        train_state = TrainState.create(
            apply_fn=policy.network.apply, params=init_params, tx=make_optimizer(config),
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        init_hstate = policy.init_hstate(num_actors)
        init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
        runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng, reward_norm_init())

        return runner_state, policy

    def make_step_fn(policy):
        """Create a JIT-compilable single update step."""

        @jax.jit
        def step_fn(runner_state, update_steps):
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                rng, act_rng = jax.random.split(rng)

                last_obs_batch = batchify(last_obs, env.agents, num_actors)
                last_done_batch = batchify(last_done, env.agents, num_actors)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                avail_actions_batch = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_actors).astype(jnp.float32))

                action, value, pi, new_hstate = policy.get_action_value_policy(
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

                env_reward = batchify(reward, env.agents, num_actors).squeeze()
                shaping_frac = jnp.array(0.0)
                combined_reward = env_reward
                if rew_shaping_horizon > 0.0 and "shaped_reward" in info:
                    shaped_actors = info["shaped_reward"].swapaxes(0, 1).reshape(-1)
                    env_steps = update_steps * env_steps_per_update
                    shaping_frac = jnp.clip(1.0 - env_steps / rew_shaping_horizon, 0.0, 1.0)
                    combined_reward = env_reward + shaping_frac * shaped_actors

                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)
                info["rew_shaping_frac"] = jnp.broadcast_to(shaping_frac, (num_actors,))
                info["combined_reward"] = combined_reward

                transition = Transition(
                    batchify(new_done, env.agents, num_actors).squeeze(),
                    action,
                    value,
                    combined_reward,
                    log_prob,
                    last_obs_batch,
                    info,
                    avail_actions_batch,
                )
                runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
                return runner_state, transition

            # The reward normaliser rides alongside the scan carry (it is only
            # updated per-update, not per-step), so split it off for the rollout.
            rew_norm_state = runner_state[6]
            core_state, traj_batch = jax.lax.scan(
                _env_step, runner_state[:6], None, config["ROLLOUT_LENGTH"]
            )

            train_state, env_state, last_obs, last_done, hstate, rng = core_state
            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_done_batch = batchify(last_done, env.agents, num_actors)
            last_avail = jax.vmap(env.get_avail_actions)(env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail, env.agents, num_actors).astype(jnp.float32))

            last_val = compute_last_value(
                policy,
                train_state.params,
                last_obs_batch,
                last_done_batch,
                last_avail_batch,
                hstate,
                num_actors,
            )
            # Normalise rewards by the running std before GAE (the critic and
            # bootstrap value learn in this same normalised return scale).
            if normalize_rewards:
                rew_norm_state = reward_norm_update(rew_norm_state, traj_batch.reward)
                traj_batch = traj_batch._replace(
                    reward=reward_norm_apply(rew_norm_state, traj_batch.reward),
                )
            advantages, targets = calculate_gae(config, traj_batch, last_val)
            train_state, loss_info, rng = run_ppo_epochs(
                config, policy, train_state, traj_batch, advantages, targets, rng, num_actors
            )

            metric = traj_batch.info
            metric["update_steps"] = update_steps
            metric["loss_total"] = loss_info.total_loss.mean()
            metric["loss_value"] = loss_info.value_loss.mean()
            metric["loss_policy"] = loss_info.policy_loss.mean()
            metric["entropy"] = loss_info.entropy.mean()
            metric["grad_norm"] = loss_info.grad_norm.mean()
            metric["approx_kl"] = loss_info.approx_kl.mean()
            metric["clip_frac"] = loss_info.clip_frac.mean()
            metric["value_mean"] = traj_batch.value.mean()

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, rew_norm_state)
            return runner_state, update_steps + 1, metric

        return step_fn

    return init, make_step_fn


def run_image_ippo(config, logger):
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    configure_training_dims(algorithm_config, env)
    num_seeds = algorithm_config["NUM_SEEDS"]
    num_updates = algorithm_config["NUM_UPDATES"]
    env_steps_per_update = algorithm_config["ROLLOUT_LENGTH"] * algorithm_config["NUM_ENVS"]
    freq_timesteps = float(algorithm_config.get("CHECKPOINT_FREQ_TIMESTEPS", 0) or 0)
    chunk_boundaries, num_ckpts, freq_updates = compute_chunk_boundaries(
        algorithm_config, num_updates, env_steps_per_update,
    )
    boundary_set = set(chunk_boundaries)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, num_seeds)

    init_fn, make_step_fn = make_train(algorithm_config, env)

    print(f"[image_ippo] NUM_UPDATES={num_updates}, NUM_SEEDS={num_seeds}, "
          f"NUM_ENVS={algorithm_config['NUM_ENVS']}")
    if freq_updates is not None:
        print(f"[image_ippo] Checkpoint cadence: every {freq_timesteps:.0f} env steps "
              f"({freq_updates} updates) -> {num_ckpts} checkpoints")
    else:
        print(f"[image_ippo] Checkpoint cadence: NUM_CHECKPOINTS={num_ckpts} evenly spaced")

    live_wandb = bool(algorithm_config.get("LIVE_WANDB_LOGGING", True))
    # Push live chunk-aggregated metrics on the checkpoint cadence so the
    # wandb dashboard updates at the same granularity training progresses at.
    # CHECKPOINT_FREQ_TIMESTEPS > 0 mirrors ja_ippo's chunking knob (lets
    # smoke runs request fine-grained live points without bumping NUM_CHECKPOINTS).
    if freq_timesteps > 0:
        live_log_interval = max(1, int(freq_timesteps / env_steps_per_update))
    else:
        live_log_interval = max(1, num_updates // max(1, num_ckpts - 1))

    # Per-checkpoint episode videos (shared path, every env). Default off so the
    # baseline outputs are unchanged unless explicitly enabled.
    save_ckpt_videos = bool(algorithm_config.get("SAVE_CKPT_VIDEOS", False))
    max_ckpt_videos = int(algorithm_config.get("MAX_CKPT_VIDEOS", num_ckpts))
    inner_env = env._env
    env_name = algorithm_config["ENV_NAME"]
    eval_max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    video_dir = None
    if save_ckpt_videos:
        import hydra
        video_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/videos"

    seed_outputs = []
    for s in range(num_seeds):
        print(f"[image_ippo] Seed {s+1}/{num_seeds}: initializing...")
        runner_state, policy = init_fn(rngs[s])
        step_fn = make_step_fn(policy)

        checkpoints = []
        all_metrics = []
        chunk_buffer = []
        update_steps = jnp.int32(0)

        print(f"[image_ippo] Seed {s+1}/{num_seeds}: compiling step fn...")
        for step in range(num_updates):
            runner_state, update_steps, metric = step_fn(runner_state, update_steps)
            all_metrics.append(metric)
            chunk_buffer.append(metric)

            if (step + 1) in boundary_set and len(checkpoints) < num_ckpts:
                checkpoints.append(runner_state[0].params)
                if save_ckpt_videos and s == 0 and len(checkpoints) <= max_ckpt_videos:
                    ckpt_idx = len(checkpoints) - 1
                    try:
                        from common.eval_media import rollout_and_log_video
                        rollout_and_log_video(
                            jax.random.PRNGKey(1000 + s * 100 + ckpt_idx),
                            inner_env, env_name, runner_state[0].params, policy,
                            eval_max_steps, tag=f"Eval/seed_{s}/ckpt_{ckpt_idx}",
                            savedir=video_dir, logger=logger,
                        )
                    except Exception as e:
                        print(f"[image_ippo] WARN: ckpt video failed ({e}); continuing.", flush=True)

            if step % max(1, num_updates // 10) == 0 or step == num_updates - 1:
                print(f"[image_ippo] Seed {s+1}/{num_seeds}: step {step+1}/{num_updates}")

            # Live wandb push at the chunk cadence. Stack the buffered per-
            # update metrics so `log_live_chunk_metrics` can mean-reduce them.
            is_chunk_end = ((step + 1) % live_log_interval == 0) or (step == num_updates - 1)
            if live_wandb and is_chunk_end and chunk_buffer:
                chunk_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *chunk_buffer)
                log_live_chunk_metrics(
                    chunk_metrics,
                    env_step=(step + 1) * env_steps_per_update,
                    seed_idx=s,
                    logger=logger,
                )
                chunk_buffer = []

        stacked_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *all_metrics)
        stacked_ckpts = jax.tree.map(lambda *xs: jnp.stack(xs), *checkpoints)

        seed_outputs.append({
            "final_params": runner_state[0].params,
            "metrics": stacked_metrics,
            "checkpoints": stacked_ckpts,
            "final_ckpt_idx": len(checkpoints),
        })

    print("[image_ippo] Training complete.")

    out = jax.tree.map(lambda *xs: jnp.stack(xs), *seed_outputs)

    use_best = bool(algorithm_config.get("USE_BEST_CKPT_FOR_EVAL", True))
    best_params = finalize_best_checkpoints(
        algorithm_config, out, chunk_boundaries, num_ckpts,
        env_steps_per_update, logger, print_prefix="image_ippo",
    )
    eval_out = {**out, "final_params": best_params} if use_best else out

    log_eval_video(algorithm_config, env, eval_out, logger)
    if num_seeds > 1:
        from evaluation.run_xp_seeds import run_xp
        xp_params = out.get("best_params", out["final_params"])
        savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        run_xp(env, policy, xp_params, algorithm_config, savedir, logger,
               jsd=False, task_name=algorithm_config.get("ENV_NAME"))
    report_basic_training_outputs(
        config,
        out,
        logger,
        scalar_keys=IMAGE_IPPO_SCALAR_KEYS,
        print_prefix="image_ippo",
    )
    return out

def log_eval_video(algorithm_config, env, out, logger):
    """Run one eval episode with final params (seed 0), render+log via the shared helper."""
    from common.eval_media import rollout_and_log_video

    rng = jax.random.PRNGKey(0)
    policy, _ = initialize_image_agent(algorithm_config, env, rng)
    final_params = jax.tree.map(lambda x: x[0], out["final_params"])
    inner_env = env._env
    env_name = algorithm_config["ENV_NAME"]
    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    try:
        rollout_and_log_video(
            jax.random.PRNGKey(42), inner_env, env_name, final_params, policy, max_steps,
            tag="Eval/episode_video", savedir=f"{savedir}/videos", logger=logger,
        )
    except Exception as e:
        print(f"[image_ippo] WARN: eval video failed ({e}); continuing.", flush=True)
