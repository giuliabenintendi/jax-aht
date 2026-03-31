'''
Image IPPO: Parameter-shared IPPO with ResNet+LSTM image architecture.

Ablation baseline for JA-IPPO. Same ResNet encoder, same LSTM, same PPO
hyperparameters — but no attention mechanism and no JSD intrinsic reward.
Uses parameter sharing (single network for both agents, like standard IPPO).
'''
import shutil

import hydra
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_image_agent
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import Transition, batchify, unbatchify, _create_minibatches


def make_train(config, env):
    """Build init and step functions for Image IPPO training.

    Returns (init_fn, step_fn) where:
      - init_fn(rng) -> (runner_state, policy) sets up network, optimizer, env
      - step_fn(runner_state, update_steps) -> (runner_state, update_steps, metric)
        runs one rollout + PPO update
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

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def init(rng):
        rng, init_rng = jax.random.split(rng)
        policy, init_params = initialize_image_agent(config, env, init_rng)

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
        runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng)

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

                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)

                transition = Transition(
                    batchify(new_done, env.agents, num_actors).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, num_actors).squeeze(),
                    log_prob,
                    last_obs_batch,
                    info,
                    avail_actions_batch,
                )
                runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_done_batch = batchify(last_done, env.agents, num_actors)
            last_avail = jax.vmap(env.get_avail_actions)(env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail, env.agents, num_actors).astype(jnp.float32))

            _, last_val, _, _ = policy.get_action_value_policy(
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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        _, value, pi, _ = policy.get_action_value_policy(
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

                train_state, (total_loss, grad_norm) = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, (total_loss, grad_norm)

            init_hstate = policy.init_hstate(num_actors)
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            (total_loss, (value_loss, policy_loss, entropy)), grad_norm = loss_info

            metric = traj_batch.info
            metric["update_steps"] = update_steps
            metric["loss_total"] = total_loss.mean()
            metric["loss_value"] = value_loss.mean()
            metric["loss_policy"] = policy_loss.mean()
            metric["entropy"] = entropy.mean()
            metric["grad_norm"] = grad_norm.mean()
            metric["value_mean"] = traj_batch.value.mean()

            rng = update_state[-1]
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, update_steps + 1, metric

        return step_fn

    return init, make_step_fn


def run_image_ippo(config, logger):
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    num_seeds = algorithm_config["NUM_SEEDS"]
    num_updates = int(algorithm_config["TOTAL_TIMESTEPS"] // algorithm_config["ROLLOUT_LENGTH"] // algorithm_config["NUM_ENVS"])
    num_ckpts = algorithm_config.get("NUM_CHECKPOINTS", 5)
    ckpt_interval = num_updates // max(1, num_ckpts - 1)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, num_seeds)

    init_fn, make_step_fn = make_train(algorithm_config, env)

    print(f"[image_ippo] NUM_UPDATES={num_updates}, NUM_SEEDS={num_seeds}, "
          f"NUM_ENVS={algorithm_config['NUM_ENVS']}")

    seed_outputs = []
    for s in range(num_seeds):
        print(f"[image_ippo] Seed {s+1}/{num_seeds}: initializing...")
        runner_state, policy = init_fn(rngs[s])
        step_fn = make_step_fn(policy)

        checkpoints = []
        all_metrics = []
        update_steps = jnp.int32(0)

        print(f"[image_ippo] Seed {s+1}/{num_seeds}: compiling step fn...")
        for step in range(num_updates):
            runner_state, update_steps, metric = step_fn(runner_state, update_steps)
            all_metrics.append(metric)

            should_ckpt = (step % ckpt_interval == 0) or (step == num_updates - 1)
            if should_ckpt and len(checkpoints) < num_ckpts:
                checkpoints.append(runner_state[0].params)

            if step % max(1, num_updates // 10) == 0 or step == num_updates - 1:
                print(f"[image_ippo] Seed {s+1}/{num_seeds}: step {step+1}/{num_updates}")

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

    log_eval_video(algorithm_config, env, out, logger)
    log_metrics(config, out, logger)
    return out


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


def log_eval_video(algorithm_config, env, out, logger):
    """Run one eval episode with final params (seed 0), log video to wandb."""
    import os
    from evaluation.vis_episodes import run_episode_with_states

    # Reconstruct policy (same for both agents — shared params)
    rng = jax.random.PRNGKey(0)
    policy, _ = initialize_image_agent(algorithm_config, env, rng)

    # Extract final params from seed 0
    final_params = jax.tree.map(lambda x: x[0], out["final_params"])

    inner_env = env._env

    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    ep_states, _, _ = run_episode_with_states(
        jax.random.PRNGKey(42), inner_env, final_params, policy,
        final_params, policy, max_steps,
        collect_attention=False,
    )
    print(f"[image_ippo] Eval episode: {len(ep_states)} frames collected")

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    video_dir = f"{savedir}/videos"
    os.makedirs(video_dir, exist_ok=True)
    video_path = f"{video_dir}/eval_final.mp4"

    env_name = algorithm_config["ENV_NAME"]
    if env_name in ("lbf", "lbf-image", "lbf-reward-shaping"):
        frames = _render_lbf_eval_frames(inner_env, ep_states)
    elif env_name == "card-game":
        from envs.card_game.rendering import render_card_game_eval_frames
        frames = render_card_game_eval_frames(ep_states, scale=32)
    else:
        from envs.overcooked.adhoc_overcooked_visualizer import AdHocOvercookedVisualizer
        viz = AdHocOvercookedVisualizer()
        viz.animate_mp4(
            [s.env_state for s in ep_states], inner_env.agent_view_size,
            filename=video_path, pixels_per_tile=32, fps=10,
        )
        logger.log_video("Eval/episode_video", video_path, commit=False)
        return

    from moviepy import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=10)
    clip.write_videofile(video_path, fps=10, codec='libx264', audio=False,
                         bitrate='8000k', preset='slow')
    logger.log_video("Eval/episode_video", video_path, commit=False)


def log_metrics(config, out, logger):
    '''Save train run output and log all metrics to wandb.'''
    train_metrics = out["metrics"]
    metric_names = get_metric_names(config["ENV_NAME"])
    train_stats = get_stats(train_metrics, metric_names)

    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    scalar_keys = [
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
        if "base_return" in train_stats and config.task["ENV_NAME"] == "overcooked-v1":
            soups = train_stats["base_return"][step, 0] / 20.0
            logger.log_item("Train/soups_delivered", soups, train_step=step, commit=False)

        for key, prefix in scalar_keys:
            if key in scalar_data:
                logger.log_item(f"{prefix}/{key}", float(scalar_data[key][step]),
                                train_step=step, commit=False)

        logger.log({}, step=step, commit=True)

        if step % print_interval == 0 or step == num_updates - 1:
            env_steps = (step + 1) * int(config.algorithm["ROLLOUT_LENGTH"]) * int(config.algorithm["NUM_ENVS"])
            pct = (step + 1) / num_updates * 100
            ret_str = "  ".join(f"{sn}={sd[step, 0]:.2f}" for sn, sd in train_stats.items())
            loss = float(scalar_data.get("loss_total", np.zeros(num_updates))[step])
            grad = float(scalar_data.get("grad_norm", np.zeros(num_updates))[step])
            extra = ""
            if "base_return" in train_stats and config.task["ENV_NAME"] == "overcooked-v1":
                soups = train_stats["base_return"][step, 0] / 20.0
                extra = f"  soups={soups:.1f}"
            print(f"[{pct:5.1f}%] step={step}/{num_updates}  env_steps={env_steps}  "
                  f"{ret_str}{extra}  loss={loss:.4f}  grad={grad:.3f}")

    logger.commit()

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
