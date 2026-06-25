"""Feature-resolution Other-Play transforms for equivariant LBF joint attention.

Under Other-Play each agent's observation is transformed by a symmetry element
g_i, so its attention map (at ResNet feature resolution) lives in the g_i frame
while food / partner positions live in the true frame. These helpers read each
agent's per-episode OP element from the wrapper state and apply the matching
feature-grid transform, so the JA mechanism can align attention with its
true-frame target (and re-transform the partner-attention feed into a consumer's
frame).

Mirror (V4) reflections are self-inverse, so the same forward feature transform
both un-transforms an agent's attention to the true frame and re-transforms a
true-frame map into an agent's frame. A no-OP state has no `per_agent_elem`, so
`read_op_elems` returns None and the mechanism falls back to identity — existing
runs are byte-identical.
"""
from __future__ import annotations

import jax.numpy as jnp

from envs.lbf.other_play import MIRROR_V4, _build_tables


def feat_transform_table(feat_h: int, feat_w: int, elements=MIRROR_V4):
    """(n_elem, feat_h*feat_w) source-gather table for transforming a flat feat map.

    Reuses the observation wrapper's `_build_tables` at feature resolution.
    """
    pix_perm, _ = _build_tables(feat_h, feat_w, elements)
    return jnp.asarray(pix_perm)


def read_op_elems(env_state, agents):
    """Per-agent OP element index from the wrapper state, or None when OP is off."""
    s = env_state
    for _ in range(8):
        if hasattr(s, "per_agent_elem"):
            return {a: s.per_agent_elem[a] for a in agents}
        s = getattr(s, "env_state", None)
        if s is None:
            break
    return None


def transform_feat_maps(maps, elem, table):
    """Apply per-actor element `elem` to feat maps `maps` (A, fh, fw) via `table`.

    `elem` is an (A,) int array. Mirror V4 is self-inverse, so the same call both
    un-transforms an agent's attention to the true frame and re-transforms a
    true-frame map into an agent's frame.
    """
    a, fh, fw = maps.shape
    flat = maps.reshape(a, fh * fw)
    perm = table[elem]  # (A, fh*fw)
    return jnp.take_along_axis(flat, perm, axis=-1).reshape(a, fh, fw)
