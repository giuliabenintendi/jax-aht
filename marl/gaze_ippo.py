"""Gaze-IPPO: image IPPO with attention-based gaze.

This is a pragmatic adaptation of "Gaze on the Prize" for the current
parameter-shared PPO codepath and is scoped to image observations.
"""
import os
import shutil
from typing import NamedTuple

import hydra
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_gaze_image_agent, _get_image_dims
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from agents.ja_utils import jsd_divergence
from common.plot_utils import get_stats, get_metric_names, plot_seed_aggregate
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.eval_logging import log_greedy_eval, log_eval_video
from marl.ppo_utils import batchify, unbatchify


class GazeTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    prev_reward: jnp.ndarray
    prev_action: jnp.ndarray
    gaze_reward: jnp.ndarray
    raw_env_reward: jnp.ndarray
    gaze_mu_x: jnp.ndarray
    gaze_mu_y: jnp.ndarray
    gaze_sigma_x: jnp.ndarray
    gaze_sigma_y: jnp.ndarray

def _create_gaze_minibatches(traj_batch, advantages, targets, return_targets,
                             init_hstate, num_actors, num_minibatches, perm_rng):
    batch = (init_hstate, traj_batch, advantages, targets, return_targets)
    permutation = jax.random.permutation(perm_rng, num_actors)
    shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(
            jnp.reshape(
                x,
                [x.shape[0], num_minibatches, -1] + list(x.shape[2:]),
            ),
            1,
            0,
        ),
        shuffled,
    )
    return minibatches

def _compute_future_returns(rewards, dones, gamma=1.0):
    def _scan_fn(carry, xs):
        reward_t, done_t = xs
        ret_t = reward_t + gamma * carry * (1.0 - done_t.astype(jnp.float32))
        return ret_t, ret_t

    _, returns = jax.lax.scan(
        _scan_fn,
        jnp.zeros_like(rewards[0]),
        (rewards, dones),
        reverse=True,
    )
    return returns
def make_train(config, env):
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
    )

    num_envs = config["NUM_ENVS"]
    num_actors = config["NUM_ACTORS"]
    gaze_beta_max = config.get("GAZE_BETA_MAX", 0.01)
    gaze_warmup_env_steps = config.get("GAZE_WARMUP_ENV_STEPS", 200_000)
    env_steps_per_update = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
    gaze_warmup_updates = gaze_warmup_env_steps / env_steps_per_update
    feed_other_attn = config.get("FEED_OTHER_ATTN", False)
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

    def init(rng):
        rng, init_rng = jax.random.split(rng)
        policy, init_params = initialize_gaze_image_agent(config, env, init_rng)

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

        init_prev_reward = jnp.zeros((num_actors,), dtype=jnp.float32)
        init_prev_action = jnp.zeros((num_actors,), dtype=jnp.float32)

        if feed_other_attn:
            init_other_attn = jnp.ones((num_actors, feat_h, feat_w)) / (feat_h * feat_w)
            runner_state = (
                train_state, env_state, obsv, init_done, init_hstate, _rng,
                init_prev_reward, init_prev_action, init_other_attn,
            )
        else:
            runner_state = (
                train_state, env_state, obsv, init_done, init_hstate, _rng,
                init_prev_reward, init_prev_action,
            )
        return runner_state, policy

    def make_step_fn(policy):
        def _augment_obs_with_attn(obs_batch, prev_other_attn):
            """Append upsampled partner gaze as a 4th image channel."""
            rgb = obs_batch.reshape(num_actors, img_h, img_w, 3)
            upsampled = jax.image.resize(
                prev_other_attn, (num_actors, img_h, img_w), method="nearest",
            )
            attn_max = jnp.max(upsampled, axis=(-2, -1), keepdims=True)
            upsampled = upsampled / jnp.maximum(attn_max, 1e-8)
            augmented = jnp.concatenate([rgb, upsampled[..., None]], axis=-1)
            return augmented.reshape(num_actors, -1)

        def _swap_and_reset_attn(attn_map, done_batch):
            """Swap gaze maps between agents, reset to uniform when episode ends."""
            attn = attn_map.squeeze(0)
            attn_0 = attn[:num_envs]
            attn_1 = attn[num_envs:]
            swapped = jnp.concatenate([attn_1, attn_0], axis=0)
            uniform = jnp.ones((feat_h, feat_w)) / (feat_h * feat_w)
            return jnp.where(done_batch[:, None, None], uniform[None], swapped)

        @jax.jit
        def step_fn(runner_state, update_steps):
            gaze_beta = jnp.minimum(
                gaze_beta_max,
                gaze_beta_max * update_steps / jnp.maximum(gaze_warmup_updates, 1.0),
            )

            def _env_step(runner_state, unused):
                if feed_other_attn:
                    (
                        train_state, env_state, last_obs, last_done, hstate, rng,
                        prev_reward, prev_action, prev_other_attn,
                    ) = runner_state
                else:
                    train_state, env_state, last_obs, last_done, hstate, rng, prev_reward, prev_action = runner_state

                rng, act_rng = jax.random.split(rng)
                last_obs_batch = batchify(last_obs, env.agents, num_actors)
                last_done_batch = batchify(last_done, env.agents, num_actors)
                last_done_mask = last_done_batch.squeeze()
                step_prev_reward = jnp.where(last_done_mask, 0.0, prev_reward)
                step_prev_action = jnp.where(last_done_mask, 0.0, prev_action)
                if feed_other_attn:
                    last_obs_batch = _augment_obs_with_attn(last_obs_batch, prev_other_attn)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                avail_actions_batch = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_actors).astype(jnp.float32)
                )

                action, value, pi, new_hstate, aux = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=last_obs_batch.reshape(1, num_actors, -1),
                    done=last_done_mask.reshape(1, num_actors),
                    avail_actions=avail_actions_batch.reshape(1, num_actors, -1),
                    hstate=hstate,
                    rng=act_rng,
                    prev_reward=step_prev_reward.reshape(1, num_actors),
                    prev_action=step_prev_action.reshape(1, num_actors),
                )

                log_prob = pi.log_prob(action)
                action = action.squeeze()
                log_prob = log_prob.squeeze()
                value = value.squeeze()
                attn_map = aux["attn_map"]

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
                r_gaze = -jsd_divergence(attn_0.squeeze(0), attn_1.squeeze(0))
                r_gaze = jax.lax.stop_gradient(r_gaze)

                reward_batch = batchify(reward, env.agents, num_actors).squeeze()
                r_gaze_batch = jnp.concatenate([r_gaze, r_gaze])
                combined_reward = reward_batch + gaze_beta * r_gaze_batch

                transition = GazeTransition(
                    batchify(new_done, env.agents, num_actors).squeeze(),
                    action,
                    value,
                    combined_reward,
                    log_prob,
                    last_obs_batch,
                    info,
                    avail_actions_batch,
                    step_prev_reward,
                    step_prev_action,
                    r_gaze_batch,
                    reward_batch,
                    aux["gaze_mu_x"].squeeze(0),
                    aux["gaze_mu_y"].squeeze(0),
                    aux["gaze_sigma_x"].squeeze(0),
                    aux["gaze_sigma_y"].squeeze(0),
                )
                if feed_other_attn:
                    new_done_batch = batchify(new_done, env.agents, num_actors).squeeze()
                    new_other_attn = _swap_and_reset_attn(attn_map, new_done_batch)
                    runner_state = (
                        train_state, new_env_state, new_obs, new_done, new_hstate,
                        rng, reward_batch, action.astype(jnp.float32), new_other_attn,
                    )
                else:
                    runner_state = (
                        train_state, new_env_state, new_obs, new_done, new_hstate,
                        rng, reward_batch, action.astype(jnp.float32),
                    )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            if feed_other_attn:
                (
                    train_state, env_state, last_obs, last_done, hstate, rng,
                    prev_reward, prev_action, prev_other_attn,
                ) = runner_state
            else:
                train_state, env_state, last_obs, last_done, hstate, rng, prev_reward, prev_action = runner_state

            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_done_batch = batchify(last_done, env.agents, num_actors)
            last_done_mask = last_done_batch.squeeze()
            step_prev_reward = jnp.where(last_done_mask, 0.0, prev_reward)
            step_prev_action = jnp.where(last_done_mask, 0.0, prev_action)
            if feed_other_attn:
                last_obs_batch = _augment_obs_with_attn(last_obs_batch, prev_other_attn)
            last_avail = jax.vmap(env.get_avail_actions)(env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail, env.agents, num_actors).astype(jnp.float32)
            )

            _, last_val, _, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs_batch.reshape(1, num_actors, -1),
                done=last_done_mask.reshape(1, num_actors),
                avail_actions=last_avail_batch.reshape(1, num_actors, -1),
                hstate=hstate,
                rng=jax.random.PRNGKey(0),
                prev_reward=step_prev_reward.reshape(1, num_actors),
                prev_action=step_prev_action.reshape(1, num_actors),
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
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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
                    init_hstate, minibatch, mb_advantages, mb_targets, _mb_returns = batch_info

                    def _loss_fn(params, minibatch, mb_advantages, mb_targets, _mb_returns):
                        _, value, pi, _, _ = policy.get_action_value_policy(
                            params=params,
                            obs=minibatch.obs,
                            done=minibatch.done,
                            avail_actions=minibatch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0),
                            prev_reward=minibatch.prev_reward,
                            prev_action=minibatch.prev_action,
                        )
                        log_prob = pi.log_prob(minibatch.action)

                        value_pred_clipped = minibatch.value + (
                            value - minibatch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - mb_targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - mb_targets)
                        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - minibatch.log_prob)
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        loss_actor1 = ratio * mb_advantages
                        loss_actor2 = jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        ) * mb_advantages
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        aux = (
                            value_loss,
                            loss_actor,
                            entropy,
                        )
                        return total_loss, aux

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, minibatch, mb_advantages, mb_targets, _mb_returns
                    )
                    grad_norm = jnp.sqrt(
                        sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, grad_norm)

                train_state, init_hstate, traj_batch, advantages, targets, return_targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_gaze_minibatches(
                    traj_batch,
                    advantages,
                    targets,
                    return_targets,
                    init_hstate,
                    num_actors,
                    config["NUM_MINIBATCHES"],
                    perm_rng,
                )
                train_state, minibatch_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, return_targets, rng)
                return update_state, minibatch_info

            init_hstate = policy.init_hstate(num_actors)
            rollout_returns = _compute_future_returns(
                traj_batch.reward,
                traj_batch.done,
                gamma=1.0,
            )
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rollout_returns, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            (total_loss, (
                value_loss,
                policy_loss,
                entropy,
            )), grad_norm = loss_info

            gaze_rew_0 = traj_batch.gaze_reward[:, :num_envs]
            jsd_values = -gaze_rew_0

            metric = traj_batch.info
            metric["update_steps"] = update_steps
            metric["gaze_beta"] = gaze_beta
            metric["jsd_mean"] = jsd_values.mean()
            metric["gaze_reward_mean"] = gaze_rew_0.mean()
            metric["gaze_mu_x_mean"] = traj_batch.gaze_mu_x.mean()
            metric["gaze_mu_y_mean"] = traj_batch.gaze_mu_y.mean()
            metric["gaze_mu_abs_mean"] = (
                jnp.abs(traj_batch.gaze_mu_x) + jnp.abs(traj_batch.gaze_mu_y)
            ).mean() / 2.0
            metric["gaze_sigma_x_mean"] = traj_batch.gaze_sigma_x.mean()
            metric["gaze_sigma_y_mean"] = traj_batch.gaze_sigma_y.mean()
            metric["raw_env_reward_mean"] = traj_batch.raw_env_reward[:, :num_envs].mean()
            metric["combined_reward_mean"] = traj_batch.reward[:, :num_envs].mean()
            metric["loss_total"] = total_loss.mean()
            metric["loss_value"] = value_loss.mean()
            metric["loss_policy"] = policy_loss.mean()
            metric["entropy"] = entropy.mean()
            metric["grad_norm"] = grad_norm.mean()
            metric["value_mean"] = traj_batch.value.mean()

            rng = update_state[-1]
            if feed_other_attn:
                runner_state = (
                    train_state, env_state, last_obs, last_done, hstate, rng,
                    prev_reward, prev_action, prev_other_attn,
                )
            else:
                runner_state = (
                    train_state, env_state, last_obs, last_done, hstate, rng,
                    prev_reward, prev_action,
                )
            return runner_state, update_steps + 1, metric

        return step_fn

    return init, make_step_fn


def run_gaze_ippo(config, logger):
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

    print(f"[gaze_ippo] NUM_UPDATES={num_updates}, NUM_SEEDS={num_seeds}, "
          f"NUM_ENVS={algorithm_config['NUM_ENVS']}")

    seed_outputs = []
    for s in range(num_seeds):
        print(f"[gaze_ippo] Seed {s+1}/{num_seeds}: initializing...")
        runner_state, policy = init_fn(rngs[s])
        step_fn = make_step_fn(policy)

        checkpoints = []
        all_metrics = []
        update_steps = jnp.int32(0)

        for step in range(num_updates):
            runner_state, update_steps, metric = step_fn(runner_state, update_steps)
            all_metrics.append(metric)
            should_ckpt = (step % ckpt_interval == 0) or (step == num_updates - 1)
            if should_ckpt and len(checkpoints) < num_ckpts:
                checkpoints.append(runner_state[0].params)
            if step % max(1, num_updates // 10) == 0 or step == num_updates - 1:
                print(f"[gaze_ippo] Seed {s+1}/{num_seeds}: step {step+1}/{num_updates}")

        stacked_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *all_metrics)
        stacked_ckpts = jax.tree.map(lambda *xs: jnp.stack(xs), *checkpoints)
        seed_outputs.append({
            "final_params": runner_state[0].params,
            "metrics": stacked_metrics,
            "checkpoints": stacked_ckpts,
            "final_ckpt_idx": len(checkpoints),
        })

    print("[gaze_ippo] Training complete.")
    out = jax.tree.map(lambda *xs: jnp.stack(xs), *seed_outputs)

    log_metrics(config, out, logger)
    _gaze_init = lambda cfg, e, r: initialize_gaze_image_agent(cfg, e, r)
    log_greedy_eval(algorithm_config, env, out, logger, init_fn=_gaze_init)
    log_eval_video(algorithm_config, env, out, logger, init_fn=_gaze_init)
    return out


def log_metrics(config, out, logger):
    """Save train run output, export CSV, and log mean+/-std to wandb."""
    import csv

    train_metrics = out["metrics"]
    metric_names = get_metric_names(config["ENV_NAME"])
    train_stats = get_stats(train_metrics, metric_names)

    algorithm_config = dict(config.algorithm)
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    rollout_length = int(algorithm_config["ROLLOUT_LENGTH"])
    num_envs = int(algorithm_config["NUM_ENVS"])

    # Save mean+/-std training curve PNGs
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

    # Scalar metrics: (metric_key, wandb_name) — matching JA-IPPO panel layout
    scalar_keys = [
        ("gaze_beta",              "JA/beta"),
        ("jsd_mean",               "JA/jsd"),
        ("raw_env_reward_mean",    "Reward/env_raw"),
        ("combined_reward_mean",   "Reward/combined_raw"),
        ("loss_total",             "Loss/total"),
        ("loss_value",             "Loss/value"),
        ("loss_policy",            "Loss/policy"),
        ("entropy",                "Loss/entropy"),
        ("grad_norm",              "Loss/grad_norm"),
        ("value_mean",             "Value/mean"),
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
            for key, _ in scalar_keys:
                if key in scalar_mean:
                    row.extend([float(scalar_mean[key][step]), float(scalar_std[key][step])])
            writer.writerow(row)

    print(f"[gaze_ippo] CSV: {csv_path} ({num_updates} updates, {num_seeds} seeds, {len(csv_header)} cols)")
    _wandb.save(csv_path, base_path=savedir)

    # --- Log to wandb ---
    print_interval = max(1, num_updates // 20)

    for step in range(num_updates):
        # Episode metrics
        for stat_name, stat_data in episode_stats_mean.items():
            logger.log_item(f"Train/{stat_name}_mean", stat_data[step, 0], train_step=step, commit=False)
            if num_seeds > 1:
                logger.log_item(f"Train/{stat_name}_std", stat_data[step, 1], train_step=step, commit=False)

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
            beta = float(scalar_mean.get("gaze_beta", np.zeros(num_updates))[step])
            loss = float(scalar_mean.get("loss_total", np.zeros(num_updates))[step])
            grad = float(scalar_mean.get("grad_norm", np.zeros(num_updates))[step])
            print(f"[{pct:5.1f}%] step={step}/{num_updates}  env_steps={env_steps}  "
                  f"{ret_str}  jsd={jsd:.4f}  beta={beta:.4f}  loss={loss:.4f}  grad={grad:.3f}")

    logger.commit()

    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
