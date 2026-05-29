"""Shared PPO building blocks for the Joint Attention trainers.

Complements `marl/ippo_core.py` (the non-attention PPO core, whose
`compute_last_value` expects a 4-tuple policy and whose `run_ppo_epochs` has no
auxiliary loss). These helpers hold the math the two JA-IPPO trainers —
`ja_ippo.py` (card game, with Other-Play) and `ja_ippo_lbf.py` (LBF) — would
otherwise duplicate: the standard clipped actor-critic losses, the global
grad-norm, and a JA-aware value bootstrap. The env-specific JA mechanism
(entity attention pooling, partner feed, shaping rewards, aux target, and the
total-loss assembly that adds the aux term) stays in each trainer.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


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
