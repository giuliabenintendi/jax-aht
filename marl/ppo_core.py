"""Shared PPO building blocks for the MARL trainers.

The numerical core every IPPO trainer reuses: training-dim derivation, the
optimizer/LR schedule, streaming reward normalization (Welford), the clipped
actor-critic losses (4-tuple and JA 5-tuple variants), GAE, and the PPO
minibatch update. Per-env mechanism logic (attention pooling, partner feed,
shaping rewards, aux loss) stays in each trainer; this module holds only the
env-agnostic math.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from marl.ppo_utils import _create_minibatches


def configure_training_dims(config, env):
    """Populate actor/update/minibatch counts in the mutable config dict."""
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
    )


def linear_schedule(config, count):
    frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
    return config["LR"] * frac


def make_optimizer(config):
    if config["ANNEAL_LR"]:
        learning_rate = lambda count: linear_schedule(config, count)
        return optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        )

    return optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )


# --------------------------------------------------------------------------- #
# Streaming reward normalization (Welford), gated by each trainer's
# NORMALIZE_REWARDS flag. Mirrors the DeepMind reference that normalised the
# combined env+intrinsic reward stream; default off so envs that never
# normalised (e.g. LBF) are unchanged unless their config opts in.
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
# PPO loss / stats containers.
# --------------------------------------------------------------------------- #
class PPOLossStats(NamedTuple):
    total_loss: jnp.ndarray
    value_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    entropy: jnp.ndarray
    grad_norm: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_frac: jnp.ndarray


class PPOLossTerms(NamedTuple):
    value_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    entropy: jnp.ndarray
    ratio: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_frac: jnp.ndarray


class PPOAuxStats(NamedTuple):
    """Per-update PPO stats including the JA auxiliary loss term."""
    total_loss: jnp.ndarray
    value_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    entropy: jnp.ndarray
    grad_norm: jnp.ndarray
    aux_loss: jnp.ndarray
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


def compute_last_value(policy, params, last_obs_batch, last_done_batch, last_avail_batch, hstate, num_actors):
    """Bootstrap the final value estimate for a rollout (non-JA 4-tuple policy)."""
    _, last_val, _, _ = policy.get_action_value_policy(
        params=params,
        obs=last_obs_batch.reshape(1, num_actors, -1),
        done=last_done_batch.reshape(1, num_actors),
        avail_actions=last_avail_batch.reshape(1, num_actors, -1),
        hstate=hstate,
        rng=jax.random.PRNGKey(0),
    )
    return last_val.squeeze()


def compute_last_value_ja(
    policy, params, last_obs_batch, last_done_batch, last_avail_batch, hstate, num_actors,
    **policy_kwargs,
):
    """Bootstrap the final value for a JA rollout.

    Unpacks the 5-tuple from `get_action_value_policy` (the non-JA
    `compute_last_value` expects a 4-tuple). `policy_kwargs` forwards
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


def calculate_gae(config, traj_batch, last_val):
    """Compute GAE advantages and value targets for a rollout batch."""
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


def run_ppo_epochs(config, policy, train_state, traj_batch, advantages, targets, rng, num_actors):
    """Run PPO minibatch updates and return updated train state plus loss stats."""
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
                value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
                ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                log_prob = pi.log_prob(traj_batch.action)
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
                policy_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()
                entropy = pi.entropy().mean()
                approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
                clip_frac = (jnp.abs(ratio - 1.0) > config["CLIP_EPS"]).mean()

                total_loss = (
                    policy_loss
                    + config["VF_COEF"] * value_loss
                    - config["ENT_COEF"] * entropy
                )
                return total_loss, (value_loss, policy_loss, entropy, approx_kl, clip_frac)

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (total_loss, (value_loss, policy_loss, entropy, approx_kl, clip_frac)), grads = grad_fn(
                train_state.params, traj_batch, advantages, targets
            )
            grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
            train_state = train_state.apply_gradients(grads=grads)
            stats = PPOLossStats(
                total_loss=total_loss,
                value_loss=value_loss,
                policy_loss=policy_loss,
                entropy=entropy,
                grad_norm=grad_norm,
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

    init_hstate = policy.init_hstate(num_actors)
    update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
    update_state, loss_info = jax.lax.scan(
        _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
    )
    return update_state[0], loss_info, update_state[-1]
