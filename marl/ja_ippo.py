"""Unified Joint-Attention IPPO trainer (all environments).

One trainer for every JA env. The env-agnostic skeleton lives here — rollout
scan, PPO update, chunked stepping, Welford reward-norm, orbax checkpointing,
best-checkpoint selection, live logging — and the per-env JA *mechanism*
(entity attention pooling, partner feed, shaping rewards, aux target/loss,
report + eval) is supplied by a small mechanism object under `agents/`,
selected by `select_mechanism`. The shared PPO math (clipped actor-critic
losses, global grad-norm, JA-aware value bootstrap) is also here and reused by
the legacy `ja_ippo_general.py` during migration.

Loop features (chunked jit stepping, orbax checkpoint folders + chunk_scores,
best-ckpt-for-eval) apply to every env. Welford reward-norm is gated by
NORMALIZE_REWARDS so envs that never normalised (e.g. LBF) are unchanged unless
their config opts in.
"""
from __future__ import annotations

import functools
import json
import os
from typing import Any, NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_ja_image_agent
from common.save_load_utils import REPO_PATH, save_train_run
from common.train_logging import log_live_chunk_metrics
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ippo_core import calculate_gae, configure_training_dims, make_optimizer
from marl.ppo_utils import _create_minibatches, batchify, unbatchify


# --------------------------------------------------------------------------- #
# Shared PPO building blocks (also imported by ja_ippo_general.py).
# --------------------------------------------------------------------------- #
class PPOLossTerms(NamedTuple):
    value_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    entropy: jnp.ndarray
    ratio: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_frac: jnp.ndarray


def ppo_actor_critic_losses(
    pi, value, *, actions, value_old, log_prob_old, gae, targets, clip_eps,
    policy_loss_type: str = "ppo",
) -> PPOLossTerms:
    """Standard clipped PPO value + policy + entropy losses.

    GAE is advantage-normalised internally. `policy_loss_type="spo"` swaps the
    clipped-min surrogate for SPO's smooth quadratic penalty (optimum at
    ratio = 1 + eps*sign(A)); any other value uses PPO clipping. Callers add
    their own auxiliary term and assemble the weighted total loss.
    """
    log_prob = pi.log_prob(actions)
    entropy = pi.entropy().mean()

    value_pred_clipped = value_old + (value - value_old).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_prob_old)
    gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
    if policy_loss_type == "spo":
        # SPO (Simple Policy Optimization): smooth quadratic penalty around
        # ratio=1 replaces PPO's clipped min, so the update can't drift far in
        # a single step.
        spo_penalty = jnp.abs(gae_norm) * jnp.square(ratio - 1.0) / (2.0 * clip_eps)
        policy_loss = -(ratio * gae_norm - spo_penalty).mean()
    else:
        loss_actor1 = ratio * gae_norm
        loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae_norm
        policy_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()

    approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
    clip_frac = (jnp.abs(ratio - 1.0) > clip_eps).mean()
    return PPOLossTerms(value_loss, policy_loss, entropy, ratio, approx_kl, clip_frac)


def global_grad_norm(grads) -> jnp.ndarray:
    """L2 norm of the full gradient pytree."""
    return jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))


def compute_last_value_ja(
    policy, params, last_obs_batch, last_done_batch, last_avail_batch, hstate, num_actors,
    **policy_kwargs,
):
    """Bootstrap the final value for a JA rollout.

    Unpacks the 5-tuple from `get_action_value_policy` (the non-JA
    `ippo_core.compute_last_value` expects a 4-tuple). `policy_kwargs` forwards
    extra inputs such as plh_actor/plh_critic when query_partner_lstm is on.
    """
    _, last_val, _, _, _ = policy.get_action_value_policy(
        params=params,
        obs=last_obs_batch.reshape(1, num_actors, -1),
        done=last_done_batch.reshape(1, num_actors),
        avail_actions=last_avail_batch.reshape(1, num_actors, -1),
        hstate=hstate,
        rng=jax.random.PRNGKey(0),
        **policy_kwargs,
    )
    return last_val.squeeze()


# --------------------------------------------------------------------------- #
# Streaming reward normalization (Welford), gated by NORMALIZE_REWARDS.
# --------------------------------------------------------------------------- #
class RewardNormState(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def reward_norm_init() -> RewardNormState:
    return RewardNormState(mean=jnp.zeros(()), var=jnp.ones(()), count=jnp.zeros(()))


def reward_norm_update(state: RewardNormState, batch: jnp.ndarray) -> RewardNormState:
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
    std = jnp.sqrt(state.var + 1e-8)
    return jnp.clip((rewards - state.mean) / std, -clip, clip)


# --------------------------------------------------------------------------- #
# Unified trainer.
# --------------------------------------------------------------------------- #
class JATransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any
    avail_actions: jnp.ndarray
    extras: Any  # env-specific pytree from the mechanism, consumed by its aux_loss


class _PPOAuxStats(NamedTuple):
    total_loss: jnp.ndarray
    value_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    entropy: jnp.ndarray
    grad_norm: jnp.ndarray
    aux_loss: jnp.ndarray


def select_mechanism(config, env):
    """Pick the per-env JA mechanism by ENV_NAME (imported lazily)."""
    env_name = config.get("ENV_NAME", "")
    if env_name == "lbf":
        from agents.lbf.ja_lbf_mechanism import LBFMechanism
        return LBFMechanism(config, env)
    if env_name == "card-game":
        from agents.card_game.ja_card_attention import CardMechanism
        return CardMechanism(config, env)
    raise NotImplementedError(
        f"JA mechanism for env '{env_name}' is not yet ported to the unified "
        f"ja_ippo trainer (Overcooked/Hanabi still run via ja_ippo_general)."
    )


def _run_ppo_epochs(config, policy, train_state, traj_batch, advantages, targets, rng, num_actors, mech):
    """PPO update; the mechanism supplies the auxiliary loss term."""

    def _update_epoch(update_state, unused):
        def _update_minbatch(train_state, batch_info):
            init_hstate, traj_batch, advantages, targets = batch_info

            def _loss_fn(params, traj_batch, gae, targets):
                hidden = policy._unpack_hstate(init_hstate)
                inputs_apply = (traj_batch.obs, traj_batch.done, traj_batch.avail_actions)
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
                total_loss = (
                    terms.policy_loss
                    + config["VF_COEF"] * terms.value_loss
                    - config["ENT_COEF"] * terms.entropy
                    + aux_coef * aux_loss
                )
                return total_loss, (terms.value_loss, terms.policy_loss, terms.entropy, aux_loss)

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (total_loss, (value_loss, policy_loss, entropy, aux_loss)), grads = grad_fn(
                train_state.params, traj_batch, advantages, targets,
            )
            grad_norm = global_grad_norm(grads)
            train_state = train_state.apply_gradients(grads=grads)
            stats = _PPOAuxStats(
                total_loss=total_loss,
                value_loss=value_loss,
                policy_loss=policy_loss,
                entropy=entropy,
                grad_norm=grad_norm,
                aux_loss=aux_loss,
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

    init_hstate = policy.init_hstate(num_actors)
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

            if normalize_rewards:
                rew_norm_state = reward_norm_update(rew_norm_state, traj_batch.reward)
                traj_batch = traj_batch._replace(
                    reward=reward_norm_apply(rew_norm_state, traj_batch.reward),
                )

            advantages, targets = calculate_gae(config, traj_batch, last_val)

            train_state, loss_info, rng = _run_ppo_epochs(
                config, policy, train_state, traj_batch, advantages, targets,
                rng, num_actors, mech,
            )

            metric = traj_batch.info
            metric["update_steps"] = update_steps
            metric["loss_total"] = loss_info.total_loss.mean()
            metric["loss_value"] = loss_info.value_loss.mean()
            metric["loss_policy"] = loss_info.policy_loss.mean()
            metric["entropy"] = loss_info.entropy.mean()
            metric["grad_norm"] = loss_info.grad_norm.mean()
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


def _select_best_per_seed_ckpt(out, chunk_boundaries):
    """Score each saved checkpoint by mean episodic return over its producing chunk.

    Returns (best_params, best_idx, per_ckpt_chunk_return). Picking by the chunk
    that produced a checkpoint approximates eval-time return without extra
    rollouts; the argmax is per seed.
    """
    metrics = out["metrics"]
    stacked_ckpts = out["checkpoints"]
    num_seeds, num_ckpts = jax.tree.leaves(stacked_ckpts)[0].shape[:2]

    returned = np.asarray(metrics["returned_episode"])
    returns = np.asarray(metrics["returned_episode_returns"])

    n_chunks = min(num_ckpts, len(chunk_boundaries))
    los = [0] + list(chunk_boundaries[:n_chunks - 1])
    his = list(chunk_boundaries[:n_chunks])

    per_ckpt_returns = np.zeros((num_seeds, num_ckpts), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(los, his)):
        m = returned[:, lo:hi]
        v = returns[:, lo:hi]
        reduce_axes = tuple(range(1, m.ndim))
        denom = np.maximum(m.sum(axis=reduce_axes), 1)
        numer = (v * m).sum(axis=reduce_axes)
        per_ckpt_returns[:, i] = numer / denom

    best_idx = per_ckpt_returns.argmax(axis=1).astype(np.int32)
    seed_arange = np.arange(num_seeds)
    best_params = jax.tree.map(lambda c: c[seed_arange, best_idx], stacked_ckpts)
    return best_params, best_idx, per_ckpt_returns


def run_ja_ippo(config, logger):
    """Unified JA-IPPO entry. Dispatches the per-env mechanism via select_mechanism."""
    algorithm_config = dict(config.algorithm)
    # Propagate COMMUNICATION into ENV_KWARGS so the env is created with it.
    if algorithm_config.get("COMMUNICATION", False):
        env_kwargs = dict(algorithm_config["ENV_KWARGS"])
        env_kwargs["communication"] = True
        algorithm_config["ENV_KWARGS"] = env_kwargs
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    mech = select_mechanism(algorithm_config, env)

    num_seeds = algorithm_config["NUM_SEEDS"]
    num_updates = int(
        algorithm_config["TOTAL_TIMESTEPS"]
        // algorithm_config["ROLLOUT_LENGTH"]
        // algorithm_config["NUM_ENVS"]
    )
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
    if freq_timesteps > 0:
        freq_updates = max(1, int(round(freq_timesteps / env_steps_per_update)))
        chunk_boundaries: list[int] = []
        b = 0
        while b < num_updates:
            b = min(b + freq_updates, num_updates)
            chunk_boundaries.append(b)
        num_ckpts = len(chunk_boundaries)
        print(f"[ja_ippo:{mech.name}] Checkpoint cadence: every {freq_timesteps:.0f} env steps "
              f"({freq_updates} updates) -> {num_ckpts} checkpoints", flush=True)
    else:
        num_ckpts = algorithm_config.get("NUM_CHECKPOINTS", 5)
        ckpt_interval = num_updates // max(1, num_ckpts - 1)
        chunk_boundaries = [min((i + 1) * ckpt_interval, num_updates) for i in range(num_ckpts)]
        if chunk_boundaries[-1] < num_updates:
            chunk_boundaries.append(num_updates)
        print(f"[ja_ippo:{mech.name}] Checkpoint cadence: NUM_CHECKPOINTS={num_ckpts} evenly spaced", flush=True)

    print(f"[ja_ippo:{mech.name}] Initializing policy and {num_seeds} seeds...", flush=True)
    policy = init_policy_fn(rngs[0])
    step_fn, chunked_step_fn = make_step_fn(policy)
    del step_fn  # chunked path is used; single-step kept for parity/debug

    live_wandb = bool(algorithm_config.get("LIVE_WANDB_LOGGING", True))

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
            seed_metrics.append(chunk_metrics)
            steps_done = chunk_end
            if len(seed_ckpts) < num_ckpts:
                seed_ckpts.append(jax.tree.map(jnp.copy, runner_state[0].params))
            print(f"[ja_ippo:{mech.name}]   step {steps_done}/{num_updates}", flush=True)
            if live_wandb:
                log_live_chunk_metrics(
                    chunk_metrics,
                    env_step=steps_done * env_steps_per_update,
                    seed_idx=seed_idx,
                    logger=logger,
                )

        all_seed_final_params.append(jax.device_get(runner_state[0].params))
        all_seed_metrics.append(jax.device_get(
            jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *seed_metrics)))
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
    best_params, best_idx, per_ckpt_returns = _select_best_per_seed_ckpt(out, chunk_boundaries)
    ckpt_env_steps = [int(b) * env_steps_per_update for b in chunk_boundaries[:num_ckpts]]
    out["best_params"] = best_params
    out["best_ckpt_idx"] = best_idx
    out["per_ckpt_chunk_return"] = per_ckpt_returns
    out["ckpt_env_steps"] = np.asarray(ckpt_env_steps, dtype=np.int64)
    best_env_steps = [ckpt_env_steps[int(i)] for i in best_idx]

    # Per-checkpoint orbax folders + chunk_scores.json under CHECKPOINT_ROOT/<run_name>/.
    savedir_for_scores = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_root_setting = str(algorithm_config.get("CHECKPOINT_ROOT", "checkpoints"))
    if not os.path.isabs(ckpt_root_setting):
        ckpt_root_setting = os.path.join(REPO_PATH, ckpt_root_setting)
    run_name = None
    if logger is not None and getattr(logger, "run", None) is not None:
        run_name = getattr(logger.run, "name", None)
    if not run_name:
        run_name = os.path.basename(savedir_for_scores.rstrip("/")) or "unnamed_run"
    ckpt_root = os.path.join(ckpt_root_setting, run_name)

    ckpt_folder_paths: list[str] = []
    if bool(algorithm_config.get("SAVE_CHECKPOINT_FOLDER", True)):
        os.makedirs(ckpt_root, exist_ok=True)
        for i in range(num_ckpts):
            params_i = jax.tree.map(lambda c, _i=i: c[:, _i], stacked_ckpts)
            ret_mean = float(per_ckpt_returns[:, i].mean())
            ckpt_name = f"ckpt_{i:02d}_ret_{ret_mean:.2f}"
            save_train_run(params_i, ckpt_root, ckpt_name)
            ckpt_folder_paths.append(os.path.join(ckpt_root, ckpt_name))
        print(f"[ja_ippo:{mech.name}] Checkpoint folder: {ckpt_root} ({num_ckpts} ckpts)", flush=True)

    scores_dir = ckpt_root if bool(algorithm_config.get("SAVE_CHECKPOINT_FOLDER", True)) else savedir_for_scores
    os.makedirs(scores_dir, exist_ok=True)
    with open(os.path.join(scores_dir, "chunk_scores.json"), "w") as _fh:
        json.dump({
            "run_name": run_name,
            "num_seeds": int(num_seeds),
            "num_ckpts": int(num_ckpts),
            "ckpt_root": ckpt_root,
            "ckpt_env_steps": ckpt_env_steps,
            "ckpt_update_boundaries": [int(b) for b in chunk_boundaries[:num_ckpts]],
            "ckpt_folder_paths": ckpt_folder_paths,
            "per_seed_per_ckpt_return": per_ckpt_returns.tolist(),
            "best_ckpt_idx_per_seed": best_idx.tolist(),
            "best_env_step_per_seed": best_env_steps,
            "best_chunk_return_per_seed": [float(per_ckpt_returns[s, best_idx[s]]) for s in range(num_seeds)],
        }, _fh, indent=2)

    mech.report(config, out, logger)

    eval_out = {**out, "final_params": best_params} if use_best else out
    mech.eval_outputs(algorithm_config, env, eval_out, logger)

    return out
