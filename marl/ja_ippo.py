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

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent
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
    normalize_rewards = config.get("NORMALIZE_REWARDS", True)
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
                        # Guard against NaN from entropy collapse (log(0) in policy)
                        total_loss = jnp.where(jnp.isnan(total_loss), 0.0, total_loss)
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

            # Combined normalization (matching paper): add intrinsic to raw, normalize together
            combined_raw = raw_env_reward + intrinsic_batch
            if normalize_rewards:
                rew_norm_state = reward_norm_update(rew_norm_state, combined_raw)
                combined = reward_norm_apply(rew_norm_state, combined_raw)
                traj_batch = traj_batch._replace(reward=combined)
            else:
                traj_batch = traj_batch._replace(reward=combined_raw)

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
    normalize_rewards = config.get("NORMALIZE_REWARDS", True)
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
                        # Guard against NaN from entropy collapse (log(0) in policy)
                        total_loss = jnp.where(jnp.isnan(total_loss), 0.0, total_loss)
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

        def _single_step(runner_state, update_steps, rew_norm_state):
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

            # Combined normalization (matching paper): add intrinsic to raw, normalize together
            combined_raw = raw_env_reward + intrinsic_batch
            if normalize_rewards:
                rew_norm_state = reward_norm_update(rew_norm_state, combined_raw)
                combined = reward_norm_apply(rew_norm_state, combined_raw)
                traj_batch = traj_batch._replace(reward=combined)
            else:
                traj_batch = traj_batch._replace(reward=combined_raw)

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
        scan_chunk = algorithm_config.get("SCAN_CHUNK_SIZE", 50)

        # Build chunk schedule: equal chunks with a smaller remainder at the end.
        # Checkpoints are saved whenever we cross a checkpoint boundary.
        chunk_sizes = []
        remaining = num_updates
        while remaining > 0:
            cs = min(scan_chunk, remaining)
            chunk_sizes.append(cs)
            remaining -= cs

        # Create policy once, then vmap state init over all seeds in parallel
        print(f"[ja_ippo] Initializing policy and {num_seeds} seeds in parallel...")
        policy = init_policy_fn(rngs[0])
        runner_state = jax.vmap(lambda rng: init_state_fn(rng, policy))(rngs)

        # Get the raw _single_step (no jit/donate decorators) for vmapping
        _, _, raw_single_step = make_step_fn(policy)

        @functools.partial(jax.jit, static_argnums=(3,))
        def vmapped_chunked_step(runner_states, update_steps_all, rew_norm_states, chunk_size):
            """Run chunk_size updates for all seeds in parallel via vmap + lax.scan."""
            def per_seed_chunk(rs, us, rns):
                def _scan_body(carry, _):
                    rs, us, rns = carry
                    rs, us, rns, metric = raw_single_step(rs, us, rns)
                    return (rs, us, rns), metric
                (rs, us, rns), metrics = jax.lax.scan(
                    _scan_body, (rs, us, rns), None, length=chunk_size)
                return rs, us, rns, metrics
            return jax.vmap(per_seed_chunk)(runner_states, update_steps_all, rew_norm_states)

        @jax.jit
        def vmapped_single_step(runner_states, update_steps_all, rew_norm_states):
            """Single update step for all seeds in parallel."""
            def per_seed(rs, us, rns):
                rs, us, rns, metric = raw_single_step(rs, us, rns)
                return rs, us, rns, metric
            return jax.vmap(per_seed)(runner_states, update_steps_all, rew_norm_states)

        # Per-seed scalars for the training loop
        update_steps = jnp.zeros(num_seeds, dtype=jnp.int32)
        rew_norm_state = jax.vmap(lambda _: reward_norm_init())(jnp.arange(num_seeds))

        checkpoints = []
        all_metrics = []
        steps_done = 0
        next_ckpt = 0

        print(f"[ja_ippo] Training {num_seeds} seeds in parallel "
              f"({len(chunk_sizes)} chunks, max {scan_chunk} steps each)...")
        for ci, cs in enumerate(chunk_sizes):
            if cs == 1:
                runner_state, update_steps, rew_norm_state, metric = vmapped_single_step(
                    runner_state, update_steps, rew_norm_state)
                metric = jax.tree.map(lambda x: x[:, None], metric)
            else:
                runner_state, update_steps, rew_norm_state, metric = vmapped_chunked_step(
                    runner_state, update_steps, rew_norm_state, cs)
            all_metrics.append(metric)
            steps_done += cs

            # Save checkpoint if we've crossed a checkpoint boundary
            # runner_state[0].params has shape (num_seeds, ...) — save the whole batch
            while next_ckpt < steps_done and len(checkpoints) < num_ckpts:
                checkpoints.append(jax.tree.map(jnp.copy, runner_state[0].params))
                next_ckpt += ckpt_interval
            if steps_done == num_updates and len(checkpoints) < num_ckpts:
                checkpoints.append(jax.tree.map(jnp.copy, runner_state[0].params))

            if ci == 0 or ci == len(chunk_sizes) - 1 or (ci + 1) % max(1, len(chunk_sizes) // 10) == 0:
                print(f"[ja_ippo] step {steps_done}/{num_updates}")

        # Metrics have shape (num_seeds, chunk_size, ...) per chunk — concat along axis 1
        stacked_metrics = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *all_metrics)
        # Checkpoints: list of (num_seeds, ...) arrays — stack along a new axis 1
        stacked_ckpts = jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *checkpoints)

        print("[ja_ippo] Training complete.")
        out = {
            "final_params": runner_state[0].params,
            "metrics": stacked_metrics,
            "checkpoints": stacked_ckpts,
            "final_ckpt_idx": len(checkpoints),
        }

    log_metrics(config, out, logger)
    log_eval_video(algorithm_config, env, out, logger)
    return out


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
        build_coverage_map, INDEX_TO_OBJECT,
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

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    for seed_idx in range(num_seeds):
        final_params = jax.tree.map(lambda x: x[seed_idx], out["final_params"])

        ep_states, attn_data = run_episode_with_states(
            jax.random.PRNGKey(42 + seed_idx), inner_env, final_params, policy,
            final_params, policy, max_steps,
            collect_attention=True,
        )
        print(f"[ja_ippo] Seed {seed_idx}: eval episode {len(ep_states)} frames collected")

        video_dir = f"{savedir}/videos/seed_{seed_idx}"
        os.makedirs(video_dir, exist_ok=True)

        # Render frames from episode states
        if env_name in ("lbf", "lbf-image", "lbf-reward-shaping"):
            frames = _render_lbf_eval_frames(inner_env, ep_states)
        else:
            from evaluation.vis_episodes import render_episode_frames
            frames = render_episode_frames(ep_states, inner_env.agent_view_size, pixels_per_tile=32)

        # Save plain eval video
        from moviepy import ImageSequenceClip
        video_path = f"{video_dir}/eval_final.mp4"
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(video_path, fps=10, codec='libx264', audio=False,
                             bitrate='8000k', preset='slow')
        tag = f"Eval/seed_{seed_idx}"
        logger.log_video(f"{tag}/episode_video", video_path, commit=False)

        # Log attention heatmaps overlaid on rendered frames
        log_attention_to_wandb(
            attn_data, logger, step=None, tag_prefix=tag, commit=False,
            frames=frames,
        )

        # Save attention overlay videos (one per agent + combined)
        attn_video_base = f"{video_dir}/eval_attention.mp4"
        make_attention_video(frames, attn_data, filename=attn_video_base, fps=10)
        logger.log_video(f"{tag}/attention_agent0", f"{video_dir}/eval_attention_agent0.mp4", commit=False)
        logger.log_video(f"{tag}/attention_agent1", f"{video_dir}/eval_attention_agent1.mp4", commit=False)
        logger.log_video(f"{tag}/attention_combined", f"{video_dir}/eval_attention_combined.mp4", commit=False)


        # Compute per-timestep attention coverage breakdown and save as JSON artifact
        if env_name in ("overcooked-v1",):
            from agents.ja_image_actor_critic import _compute_resnet_output_dims
            from agents.initialize_agents import _get_image_dims
            import json
            import numpy as np

            img_h, img_w, _ = _get_image_dims(env)
            feat_h, feat_w = _compute_resnet_output_dims(
                img_h, img_w,
                stride=algorithm_config.get("CONV_STRIDE", 2),
                kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
                padding=algorithm_config.get("CONV_PADDING", "SAME"),
                num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
            )

            attn_threshold = 0.05
            n_steps = min(len(attn_data["agent_0"]), len(ep_states) - 1)
            coverage_data = {}

            for agent_name in ("agent_0", "agent_1"):
                maps = attn_data[agent_name]
                agent_steps = []

                for t in range(n_steps):
                    attn = np.array(maps[t]).squeeze()
                    coverage = build_coverage_map(ep_states[t], feat_h, feat_w)

                    # Filter cells above threshold
                    spots = []
                    for r in range(feat_h):
                        for c in range(feat_w):
                            val = float(attn[r, c])
                            if val >= attn_threshold:
                                cov = coverage[r, c]
                                # Only include nonzero categories
                                cov_dict = {
                                    INDEX_TO_OBJECT[i]: round(float(cov[i]), 3)
                                    for i in range(len(INDEX_TO_OBJECT))
                                    if cov[i] > 0.01
                                }
                                spots.append({
                                    "attn": round(val, 4),
                                    "coverage": cov_dict,
                                })

                    spots.sort(key=lambda s: -s["attn"])
                    agent_steps.append({"t": t, "spots": spots})

                coverage_data[agent_name] = agent_steps

            # Save JSON to run directory
            json_path = f"{video_dir}/attention_coverage.json"
            with open(json_path, "w") as f:
                json.dump(coverage_data, f, indent=2)
            print(f"[ja_ippo] Seed {seed_idx} attention coverage saved to {json_path}")


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

    num_seeds = train_metrics["returned_episode"].shape[0]
    num_updates = train_metrics["returned_episode"].shape[1]

    # Per-seed episode stats → mean across seeds (shape: [updates, 2] where [:,0]=mean [:,1]=std)
    episode_stats_mean = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

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
        # Episode metrics (mean±std across seeds)
        for stat_name, stat_data in episode_stats_mean.items():
            logger.log_item(f"Train/{stat_name}_mean", stat_data[step, 0], train_step=step, commit=False)
            logger.log_item(f"Train/{stat_name}_std", stat_data[step, 1], train_step=step, commit=False)
        if "base_return" in episode_stats_mean and config.task["ENV_NAME"] == "overcooked-v1":
            soups = episode_stats_mean["base_return"][step, 0] / 20.0
            logger.log_item("Train/soups_delivered", soups, train_step=step, commit=False)

        # Scalar metrics (mean±std across seeds)
        for key, wandb_name in scalar_keys:
            if key in scalar_mean:
                logger.log_item(f"{wandb_name}/mean", float(scalar_mean[key][step]),
                                train_step=step, commit=False)
                logger.log_item(f"{wandb_name}/std", float(scalar_std[key][step]),
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
