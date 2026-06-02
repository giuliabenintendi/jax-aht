"""Streaming reward normalization (Welford), shared by the image and JA trainers.

Gated by each trainer's NORMALIZE_REWARDS flag. Mirrors the DeepMind reference
that normalised the combined env+intrinsic reward stream; default off so envs
that never normalised (e.g. LBF) are unchanged unless their config opts in.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


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
