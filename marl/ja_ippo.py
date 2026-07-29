"""Unified Joint-Attention IPPO trainer (all environments).

This is the trainer behind every result in the paper: IPPO, OP, HE IPPO, the
Lee et al. (2021) baseline and MATE all run through it, differing only in config
flags and in which mechanism `select_mechanism` returns. It implements Algorithm 1
(MATE): the rollout scan reframes and feeds the partner's attention map each step
(lines 5-6), and `_run_ppo_epochs` assembles Eq. 11's objective (line 14).

One trainer for every JA env. The env-agnostic skeleton lives here — rollout
scan, PPO update, chunked stepping, Welford reward-norm, orbax checkpointing,
best-checkpoint selection, live logging — and the per-env JA *mechanism*
(entity attention pooling, partner feed, shaping rewards, aux target/loss,
report + eval) is supplied by a small mechanism object under `agents/`,
selected by `select_mechanism`. The shared PPO math (clipped actor-critic
losses, global grad-norm, JA-aware value bootstrap) lives here too.

Loop features (chunked jit stepping, orbax checkpoint folders + chunk_scores,
best-ckpt-for-eval) apply to every env. Welford reward-norm is gated by
NORMALIZE_REWARDS so envs that never normalised (e.g. LBF) are unchanged unless
their config opts in.
"""
from __future__ import annotations

import functools
from typing import Any, NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_ja_image_agent
from common.train_logging import log_live_chunk_metrics
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.checkpointing import compute_chunk_boundaries, finalize_best_checkpoints
from marl.ppo_core import (
    PPOAuxStats,
    calculate_gae,
    compute_last_value_ja,
    configure_training_dims,
    global_grad_norm,
    make_optimizer,
    ppo_actor_critic_losses,
    reward_norm_apply,
    reward_norm_init,
    reward_norm_update,
)
from marl.ppo_utils import _create_minibatches, batchify, unbatchify


# --------------------------------------------------------------------------- #
# Unified trainer.
# --------------------------------------------------------------------------- #
class JATransition(NamedTuple):
    done: jnp.ndarray  # post-step done_t; masks the GAE bootstrap
    prev_done: jnp.ndarray  # pre-step done_{t-1}; the LSTM reset flag paired with obs_t when acting
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any
    avail_actions: jnp.ndarray
    extras: Any  # env-specific pytree from the mechanism, consumed by its aux_loss


def select_mechanism(config, env):
    """Pick the per-env JA mechanism by ENV_NAME (imported lazily)."""
    env_name = config.get("ENV_NAME", "")
    if env_name == "lbf":
        from agents.lbf.ja_lbf_mate import LBFDenseObjectOccupancyMechanism
        return LBFDenseObjectOccupancyMechanism(config, env)
    if env_name == "card-game":
        from agents.card_game.ja_card_attention import CardMechanism
        return CardMechanism(config, env)
    if env_name == "overcooked-v2" and config.get("JA_OBJECT_OCCUPANCY", False):
        from agents.overcooked_v2.ja_overcooked_v2_mate import (
            OvercookedV2DenseObjectOccupancyMechanism,
        )
        return OvercookedV2DenseObjectOccupancyMechanism(config, env)
    if env_name == "overcooked-v2":
        # No MATE flags: the IPPO / OP / HE IPPO baselines.
        from agents.baseline_mechanism import BaselineMechanism
        return BaselineMechanism(config, env)
    raise NotImplementedError(f"No JA mechanism registered for env '{env_name}'.")


def _run_ppo_epochs(config, policy, train_state, traj_batch, advantages, targets, init_hstate, rng, num_actors, mech):
    """PPO update; the mechanism supplies the auxiliary loss term.

    `init_hstate` must be the actors' hidden state at the START of the rollout:
    episodes span rollout cuts, so replaying from a zero state would condition
    the recomputed action distributions differently than during acting.
    """

    def _update_epoch(update_state, unused):
        def _update_minbatch(train_state, batch_info):
            init_hstate, traj_batch, advantages, targets = batch_info

            def _loss_fn(params, traj_batch, gae, targets):
                hidden = policy._unpack_hstate(init_hstate)
                inputs_apply = (traj_batch.obs, traj_batch.prev_done, traj_batch.avail_actions)
                _, pi, value, attn_map_apply = policy.network.apply(params, hidden, inputs_apply)

                aux_coef, aux_loss = mech.aux_loss(attn_map_apply, traj_batch, config)
                terms = ppo_actor_critic_losses(
                    pi, value,
                    actions=traj_batch.action,
                    value_old=traj_batch.value,
                    log_prob_old=traj_batch.log_prob,
                    gae=gae, targets=targets,
                    clip_eps=config["CLIP_EPS"],
                    policy_loss_type=config.get("POLICY_LOSS_TYPE", "ppo"),
                )
                # Eq. 11: L = L_IPPO + lambda_MATE * L_MATE. The first three terms are
                # L_IPPO; aux_coef is lambda_MATE and aux_loss is L_MATE (Eq. 10),
                # both supplied by the env's mechanism. Baselines return coef 0.
                total_loss = (
                    terms.policy_loss
                    + config["VF_COEF"] * terms.value_loss
                    - config["ENT_COEF"] * terms.entropy
                    + aux_coef * aux_loss
                )
                return total_loss, (
                    terms.value_loss,
                    terms.policy_loss,
                    terms.entropy,
                    aux_loss,
                    terms.approx_kl,
                    terms.clip_frac,
                )

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (
                total_loss,
                (value_loss, policy_loss, entropy, aux_loss, approx_kl, clip_frac),
            ), grads = grad_fn(
                train_state.params, traj_batch, advantages, targets,
            )
            grad_norm = global_grad_norm(grads)
            train_state = train_state.apply_gradients(grads=grads)
            stats = PPOAuxStats(
                total_loss=total_loss,
                value_loss=value_loss,
                policy_loss=policy_loss,
                entropy=entropy,
                grad_norm=grad_norm,
                aux_loss=aux_loss,
                approx_kl=approx_kl,
                clip_frac=clip_frac,
            )
            return train_state, stats

        train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
        rng, perm_rng = jax.random.split(rng)
        minibatches = _create_minibatches(
            traj_batch, advantages, targets, init_hstate,
            num_actors, config["NUM_MINIBATCHES"], perm_rng,
        )
        train_state, stats = jax.lax.scan(_update_minbatch, train_state, minibatches)
        update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
        return update_state, stats

    update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
    update_state, loss_info = jax.lax.scan(
        _update_epoch, update_state, None, config["UPDATE_EPOCHS"],
    )
    return update_state[0], loss_info, update_state[-1]


def make_train(config, env, mech):
    configure_training_dims(config, env)
    num_envs = config["NUM_ENVS"]
    num_actors = config["NUM_ACTORS"]
    rollout_length = config["ROLLOUT_LENGTH"]
    normalize_rewards = bool(config.get("NORMALIZE_REWARDS", False))

    config = dict(config)
    config["JA_ENTITY_FEED_DIM"] = mech.entity_feed_dim()

    def init_policy(rng):
        rng, init_rng = jax.random.split(rng)
        policy, _ = initialize_ja_image_agent(config, env, init_rng)
        return policy

    def init_state(rng, policy):
        rng, init_rng = jax.random.split(rng)
        _, init_params = initialize_ja_image_agent(config, env, init_rng)
        train_state = TrainState.create(
            apply_fn=policy.network.apply, params=init_params, tx=make_optimizer(config),
        )
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = policy.init_hstate(num_actors)
        init_done = {k: jnp.zeros((num_envs,), dtype=bool) for k in env.agents + ["__all__"]}
        carry = mech.init_carry(num_actors)
        return (train_state, env_state, obsv, init_done, init_hstate, _rng, carry)

    def make_step_fn(policy):
        def _single_step(runner_state, update_steps, rew_norm_state):
            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, last_done, hstate, rng, carry) = runner_state
                rng, act_rng = jax.random.split(rng)

                last_obs_batch = batchify(last_obs, env.agents, num_actors)
                last_done_batch = batchify(last_done, env.agents, num_actors)
                obs_aug = mech.augment_obs(last_obs_batch, carry)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                avail_actions_batch = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_actors).astype(jnp.float32),
                )

                action, value, pi, new_hstate, attn_map = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=obs_aug.reshape(1, num_actors, -1),
                    done=last_done_batch.reshape(1, num_actors),
                    avail_actions=avail_actions_batch.reshape(1, num_actors, -1),
                    hstate=hstate,
                    rng=act_rng,
                )
                log_prob = pi.log_prob(action).squeeze()
                action = action.squeeze()
                value = value.squeeze()

                env_act = unbatchify(action, env.agents, num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                new_obs, new_env_state, reward, new_done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0),
                )(rng_step, env_state, env_act)

                done_actors = batchify(new_done, env.agents, num_actors).squeeze().astype(bool)
                env_reward = batchify(reward, env.agents, num_actors).squeeze()

                shaped_reward, new_carry, extras = mech.step(
                    attn_map=attn_map, env_state=env_state, new_env_state=new_env_state,
                    action=action, env_reward=env_reward, info=info,
                    done_actors=done_actors, carry=carry,
                    num_actors=num_actors, update_steps=update_steps,
                )

                transition = JATransition(
                    done=done_actors,
                    prev_done=last_done_batch.squeeze().astype(bool),
                    action=action,
                    value=value,
                    reward=shaped_reward,
                    log_prob=log_prob,
                    obs=obs_aug,
                    info=jax.tree.map(lambda x: x.reshape((num_actors,)), info),
                    avail_actions=avail_actions_batch,
                    extras=extras,
                )
                runner_state = (
                    train_state, new_env_state, new_obs, new_done, new_hstate, rng, new_carry,
                )
                return runner_state, transition

            # Snapshot the hidden state before the rollout: the PPO replay
            # must unroll the LSTM from here, not from zeros.
            rollout_start_hstate = runner_state[4]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, rollout_length,
            )

            (train_state, env_state, last_obs, last_done, hstate, rng, carry) = runner_state

            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_done_batch = batchify(last_done, env.agents, num_actors)
            last_avail = jax.vmap(env.get_avail_actions)(env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail, env.agents, num_actors).astype(jnp.float32),
            )
            last_obs_aug = mech.augment_obs(last_obs_batch, carry)
            last_val = compute_last_value_ja(
                policy, train_state.params,
                last_obs_aug, last_done_batch, last_avail_batch, hstate, num_actors,
            )

            postprocess = getattr(mech, "postprocess_trajectory", None)
            if postprocess is not None:
                traj_batch = postprocess(traj_batch, config, update_steps)

            if normalize_rewards:
                rew_norm_state = reward_norm_update(rew_norm_state, traj_batch.reward)
                traj_batch = traj_batch._replace(
                    reward=reward_norm_apply(rew_norm_state, traj_batch.reward),
                )

            advantages, targets = calculate_gae(config, traj_batch, last_val)

            train_state, loss_info, rng = _run_ppo_epochs(
                config, policy, train_state, traj_batch, advantages, targets,
                rollout_start_hstate, rng, num_actors, mech,
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
            metric.update(mech.rollout_metrics(traj_batch, loss_info))

            runner_state = (
                train_state, env_state, last_obs, last_done, hstate, rng, carry,
            )
            return runner_state, update_steps + 1, rew_norm_state, metric

        @functools.partial(jax.jit, donate_argnums=(0, 2))
        def step_fn(runner_state, update_steps, rew_norm_state):
            return _single_step(runner_state, update_steps, rew_norm_state)

        @functools.partial(jax.jit, static_argnums=(3,), donate_argnums=(0, 2))
        def chunked_step_fn(runner_state, update_steps, rew_norm_state, chunk_size):
            def _scan_body(carry, _):
                rs, us, rns = carry
                rs, us, rns, metric = _single_step(rs, us, rns)
                return (rs, us, rns), metric

            (runner_state, update_steps, rew_norm_state), metrics = jax.lax.scan(
                _scan_body, (runner_state, update_steps, rew_norm_state), None, length=chunk_size,
            )
            return runner_state, update_steps, rew_norm_state, metrics

        return step_fn, chunked_step_fn

    return init_policy, init_state, make_step_fn


def run_ja_ippo(config, logger):
    """Unified JA-IPPO entry. Dispatches the per-env mechanism via select_mechanism."""
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    mech = select_mechanism(algorithm_config, env)

    configure_training_dims(algorithm_config, env)
    num_seeds = algorithm_config["NUM_SEEDS"]
    num_updates = algorithm_config["NUM_UPDATES"]
    print(
        f"[ja_ippo:{mech.name}] NUM_UPDATES={num_updates}, NUM_SEEDS={num_seeds}, "
        f"NUM_ENVS={algorithm_config['NUM_ENVS']}",
        flush=True,
    )

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, num_seeds)

    init_policy_fn, init_state_fn, make_step_fn = make_train(algorithm_config, env, mech)

    env_steps_per_update = int(algorithm_config["ROLLOUT_LENGTH"]) * int(algorithm_config["NUM_ENVS"])
    freq_timesteps = float(algorithm_config.get("CHECKPOINT_FREQ_TIMESTEPS", 0) or 0)
    chunk_boundaries, num_ckpts, freq_updates = compute_chunk_boundaries(
        algorithm_config, num_updates, env_steps_per_update,
    )
    if freq_updates is not None:
        print(f"[ja_ippo:{mech.name}] Checkpoint cadence: every {freq_timesteps:.0f} env steps "
              f"({freq_updates} updates) -> {num_ckpts} checkpoints", flush=True)
    else:
        print(f"[ja_ippo:{mech.name}] Checkpoint cadence: NUM_CHECKPOINTS={num_ckpts} evenly spaced", flush=True)

    print(f"[ja_ippo:{mech.name}] Initializing policy and {num_seeds} seeds...", flush=True)
    policy = init_policy_fn(rngs[0])
    step_fn, chunked_step_fn = make_step_fn(policy)
    del step_fn  # chunked path is used; single-step kept for parity/debug

    live_wandb = bool(algorithm_config.get("LIVE_WANDB_LOGGING", True))

    # Per-checkpoint episode videos via the shared helper. Default off so the
    # paper pipeline is byte-identical unless explicitly enabled.
    save_ckpt_videos = bool(algorithm_config.get("SAVE_CKPT_VIDEOS", False))
    max_ckpt_videos = int(algorithm_config.get("MAX_CKPT_VIDEOS", num_ckpts))
    ckpt_video_every = int(algorithm_config.get("CKPT_VIDEO_EVERY", 0))
    inner_env = getattr(env, "_env", env)
    env_name = algorithm_config["ENV_NAME"]
    eval_max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    video_dir = None
    if save_ckpt_videos:
        video_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/videos"

    all_seed_metrics = []
    all_seed_ckpts = []
    all_seed_final_params = []

    for seed_idx in range(num_seeds):
        runner_state = init_state_fn(rngs[seed_idx], policy)
        update_steps = jnp.zeros((), dtype=jnp.int32)
        rew_norm_state = reward_norm_init()

        seed_metrics = []
        seed_ckpts = []
        steps_done = 0

        print(f"[ja_ippo:{mech.name}] Seed {seed_idx}/{num_seeds}: training {num_updates} steps...", flush=True)
        for chunk_end in chunk_boundaries:
            chunk_size = chunk_end - steps_done
            if chunk_size <= 0:
                continue
            runner_state, update_steps, rew_norm_state, chunk_metrics = chunked_step_fn(
                runner_state, update_steps, rew_norm_state, chunk_size,
            )
            # Keep historical metrics on host so GPU memory does not grow per chunk.
            host_chunk_metrics = jax.device_get(chunk_metrics)
            del chunk_metrics
            seed_metrics.append(host_chunk_metrics)
            steps_done = chunk_end
            if len(seed_ckpts) < num_ckpts:
                seed_ckpts.append(jax.tree.map(jnp.copy, runner_state[0].params))
                ckpt_idx = len(seed_ckpts) - 1
                # CKPT_VIDEO_EVERY>0 renders at a fixed cadence (ckpt 0, N, 2N, ...) so
                # videos span the whole run; else cap to the first MAX_CKPT_VIDEOS.
                want_ckpt_video = (
                    ckpt_idx % ckpt_video_every == 0 if ckpt_video_every > 0
                    else len(seed_ckpts) <= max_ckpt_videos
                )
                if save_ckpt_videos and seed_idx == 0 and want_ckpt_video:
                    tag = f"Eval/seed_{seed_idx}/ckpt_{ckpt_idx}"
                    ckpt_video = getattr(mech, "log_ckpt_video", None)
                    try:
                        if ckpt_video is not None:
                            ckpt_video(algorithm_config, env, runner_state[0].params,
                                       policy, tag, video_dir, logger)
                        else:
                            from common.eval_media import rollout_and_log_video
                            rollout_and_log_video(
                                jax.random.PRNGKey(7000 + seed_idx * 100 + ckpt_idx),
                                inner_env, env_name, runner_state[0].params, policy,
                                eval_max_steps, tag=tag, savedir=video_dir, logger=logger,
                            )
                    except Exception as e:
                        print(f"[ja_ippo:{mech.name}] WARN: ckpt video failed ({e}); continuing.", flush=True)
            print(f"[ja_ippo:{mech.name}]   step {steps_done}/{num_updates}", flush=True)
            if live_wandb:
                log_live_chunk_metrics(
                    host_chunk_metrics,
                    env_step=steps_done * env_steps_per_update,
                    seed_idx=seed_idx,
                    logger=logger,
                    mech_scalar_keys=getattr(mech, "scalar_keys", None),
                )

        all_seed_final_params.append(jax.device_get(runner_state[0].params))
        all_seed_metrics.append(
            jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *seed_metrics))
        all_seed_ckpts.append(jax.device_get(
            jax.tree.map(lambda *xs: jnp.stack(xs), *seed_ckpts)))

    stacked_params = jax.tree.map(lambda *xs: np.stack(xs), *all_seed_final_params)
    stacked_metrics = jax.tree.map(lambda *xs: np.stack(xs), *all_seed_metrics)
    stacked_ckpts = jax.tree.map(lambda *xs: np.stack(xs), *all_seed_ckpts)

    print(f"[ja_ippo:{mech.name}] Training complete.", flush=True)
    out = {
        "final_params": stacked_params,
        "metrics": stacked_metrics,
        "checkpoints": stacked_ckpts,
        "final_ckpt_idx": num_ckpts,
    }

    use_best = bool(algorithm_config.get("USE_BEST_CKPT_FOR_EVAL", True))
    best_params = finalize_best_checkpoints(
        algorithm_config, out, chunk_boundaries, num_ckpts,
        env_steps_per_update, logger, print_prefix=f"ja_ippo:{mech.name}",
    )

    mech.report(config, out, logger)

    eval_out = {**out, "final_params": best_params} if use_best else out
    mech.eval_outputs(algorithm_config, env, eval_out, logger)

    if num_seeds > 1:
        from evaluation.run_xp_seeds import run_xp
        xp_params = out.get("best_params", out["final_params"])
        savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        run_xp(env, policy, xp_params, algorithm_config, savedir, logger,
               jsd=True, task_name=algorithm_config.get("ENV_NAME"))

    return out
