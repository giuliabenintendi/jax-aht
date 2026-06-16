"""Target-agnostic future-occupancy maps for generalizable Joint Attention.

This JA mechanism replaces "predict which discrete target the partner picks"
(per-entity pooling = hardcoded targets) with "predict where the partner will
*be*": a gamma-discounted FIRST-occupancy heatmap over the spatial feature grid,
the same shape as the JA attention map. It carries no entity/target knowledge —
only the cells an agent visits — so it ports across envs unchanged.

The label is self-supervised and free: one parameter-shared actor rolls out both
agents, so each agent's future trajectory is already in the batch. A single
reverse scan over the time axis turns recorded grid positions into the
discounted first-occupancy `O_t[cell] = gamma_occ ** (first step >= t at which
the agent reaches `cell`)`, normalised to a distribution.

The two JA shaping rewards are plain distribution overlaps (no centroids, no
divergence/logs):
  R_partner = sum_cells( attention_i  *  future_occupancy_partner )   # "attend where the partner is heading"
  R_self    = sum_cells( future_occupancy_i  *  attention_i )         # "move where you attend"
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def grid_pos_to_feat_index(pos_xy, tile_size, feat_h, feat_w, img_h, img_w):
    """Map integer grid positions (x, y) to a flat feature-cell index.

    Mirrors `per_fruit_attn`'s tile-centre -> feature-cell mapping so occupancy
    lives on the same grid as the attention map. `pos_xy[..., 0]` is the column
    (x), `pos_xy[..., 1]` the row (y).
    """
    centre_r = pos_xy[..., 1] * tile_size + tile_size // 2
    centre_c = pos_xy[..., 0] * tile_size + tile_size // 2
    fr = jnp.clip(centre_r * feat_h // img_h, 0, feat_h - 1).astype(jnp.int32)
    fc = jnp.clip(centre_c * feat_w // img_w, 0, feat_w - 1).astype(jnp.int32)
    return fr * feat_w + fc


def discounted_first_occupancy(pos_traj_xy, gamma_occ, tile_size,
                               feat_h, feat_w, img_h, img_w, done=None):
    """Per-step discounted first-occupancy heatmaps from a position trajectory.

    `pos_traj_xy`: (T, A, 2) int grid positions over a rollout (A = num actors).
    Returns `O`: (T, A, feat_h*feat_w) float, each row a distribution (sums to 1)
    giving, from step t onward, gamma_occ ** (steps until the cell is first
    reached) — the current cell scores 1.

    Implemented as a single reverse scan: `M_t = where(at cell now, 1, gamma*M_{t+1})`.
    First-occupancy (reset-to-1 at the visited cell), not cumulative occupancy,
    so it highlights the destination/path rather than where the agent loiters.
    If `done` is provided, terminal transitions do not include future mass from
    the auto-reset episode that follows.
    """
    cells = grid_pos_to_feat_index(pos_traj_xy, tile_size, feat_h, feat_w, img_h, img_w)  # (T, A)
    num_cells = feat_h * feat_w
    onehot = jax.nn.one_hot(cells, num_cells, dtype=jnp.float32)          # (T, A, C)
    if done is None:
        done = jnp.zeros(cells.shape, dtype=bool)

    def _scan(m_next, x_t):
        oh_t, done_t = x_t
        m_next = jnp.where(done_t[..., None], 0.0, m_next)
        m_t = jnp.where(oh_t > 0.0, 1.0, gamma_occ * m_next)
        return m_t, m_t

    init = jnp.zeros(onehot.shape[1:], dtype=jnp.float32)                 # (A, C)
    _, m = jax.lax.scan(_scan, init, (onehot, done), reverse=True)         # (T, A, C)
    return m / (m.sum(axis=-1, keepdims=True) + 1e-8)


def overlap(p, q):
    """Distribution overlap sum_cells(p * q) over the last axis. (..., C) -> (...)."""
    return (p * q).sum(axis=-1)


## Tests

def test_first_occupancy_recursion():
    # 1-D, 3 cells, agent walks 0 -> 1 -> 2, gamma=0.5. Map cell k to grid x=k on
    # a tile_size=1 / feat=img grid so feat index == x.
    pos = jnp.array([[[0, 0]], [[1, 0]], [[2, 0]]], dtype=jnp.int32)  # (T=3, A=1, 2)
    o = discounted_first_occupancy(pos, 0.5, tile_size=1, feat_h=1, feat_w=3, img_h=1, img_w=3)
    # Unnormalised FR from t=0 is [1, .5, .25]; normalised by 1.75.
    expected0 = jnp.array([1.0, 0.5, 0.25]) / 1.75
    assert jnp.allclose(o[0, 0], expected0, atol=1e-6)
    # From t=1: [0, 1, .5] / 1.5; current cell (1) is the mode.
    assert jnp.allclose(o[1, 0], jnp.array([0.0, 1.0, 0.5]) / 1.5, atol=1e-6)
    assert int(jnp.argmax(o[1, 0])) == 1


def test_overlap_agreement():
    p = jnp.array([0.0, 0.3, 0.7])
    assert jnp.allclose(overlap(p, jnp.array([0.0, 0.2, 0.8])), 0.62, atol=1e-6)  # agree on cell 2
    assert jnp.allclose(overlap(p, jnp.array([0.8, 0.2, 0.0])), 0.06, atol=1e-6)  # disagree
