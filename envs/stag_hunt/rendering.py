"""Eval-frame rendering for the Stag Hunt env.

Renders the ego-centric agent views (ego red, partner blue, stags green, plants
yellow, walls grey) from a rollout's states, for the per-checkpoint videos.
"""
from __future__ import annotations

import numpy as np


def render_stag_hunt_ego_frames(
    inner_env, ep_states, upscale: int = 8
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render per-agent ego-view frames from a rollout.

    `ep_states` is the list of `WrappedEnvState` returned by
    `run_episode_with_states` (so `ws.env_state` is the `StagHuntState`).
    Returns `(frames_agent0, frames_agent1)`, each a list of uint8 `(H, W, 3)`
    arrays upscaled by `upscale` (the raw tile render is only a few px/tile).
    """
    frames0: list[np.ndarray] = []
    frames1: list[np.ndarray] = []
    for ws in ep_states:
        state = ws.env_state
        img0 = np.asarray(inner_env._render_agent_view(state, 0)).astype(np.uint8)
        img1 = np.asarray(inner_env._render_agent_view(state, 1)).astype(np.uint8)
        if upscale > 1:
            img0 = np.repeat(np.repeat(img0, upscale, axis=0), upscale, axis=1)
            img1 = np.repeat(np.repeat(img1, upscale, axis=0), upscale, axis=1)
        frames0.append(img0)
        frames1.append(img1)
    return frames0, frames1
