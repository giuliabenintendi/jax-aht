"""Image/feature geometry and state helpers for Stag Hunt joint attention.

Mirrors `agents/lbf/ja_lbf_attention.py` but for the Stag Hunt env, which stores
agent positions as `[x, y]` (so no row/col swap is needed downstream).
"""
from __future__ import annotations

import jax.numpy as jnp


def stag_hunt_attention_ctx(config, env) -> dict:
    """Image/feature-map geometry needed by the future-occupancy mechanism.

    Returns `{'tile_size', 'feat_h', 'feat_w', 'img_h', 'img_w'}`.
    """
    from agents.initialize_agents import _get_image_dims
    from agents.ja_actor_critic import _compute_resnet_output_dims
    from envs.base_env import get_inner_env

    img_h, img_w, _ = _get_image_dims(env)
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w,
        stride=config.get("CONV_STRIDE", 2),
        kernel_size=config.get("CONV_KERNEL_SIZE", 3),
        padding=config.get("CONV_PADDING", "SAME"),
        num_blocks=config.get("CONV_NUM_BLOCKS", 4),
    )
    inner = get_inner_env(env)
    return {
        "tile_size": int(inner.tile_size),
        "feat_h": feat_h,
        "feat_w": feat_w,
        "img_h": img_h,
        "img_w": img_w,
    }


def _unwrap_stag_hunt_state(state):
    """Peel wrapper layers (LogWrapper / WrappedEnvState) to the StagHuntState."""
    s = state
    for _ in range(3):
        if hasattr(s, "agent_pos"):
            return s
        s = getattr(s, "env_state", None)
        if s is None:
            break
    raise AttributeError("Could not locate StagHuntState (no agent_pos found).")


def agent_positions_from_log_state(log_state):
    """Extract `(num_envs, num_agents, 2)` agent positions as `[x, y]`.

    Works for both LogWrapper-wrapped (training) and raw (eval) state shapes.
    """
    return _unwrap_stag_hunt_state(log_state).agent_pos


def as_spatial_attention(attn_map):
    """Return spatial attention with shape `(..., feat_h, feat_w)` (identity on rank-4)."""
    if attn_map.ndim == 4:
        return attn_map
    raise ValueError(f"Unexpected attention map rank: {attn_map.ndim}")
