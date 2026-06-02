"""Single dispatch from `env_name` to its eval-frame renderer.

Replaces the per-trainer `if env_name == ...` ladder. `get_eval_frames` returns a
list of uint8 `(H, W, 3)` frames (one per timestep) for any registered env.

Overcooked is the one exception: its `AdHocOvercookedVisualizer.animate_mp4` writes
the mp4 inline rather than returning frames, so it is handled directly in
`common.eval_media.render_and_log_video` and is intentionally NOT in this table.
"""
from __future__ import annotations

import numpy as np


def _composite_per_agent(frames_per_agent: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Stack per-agent ego frames side by side into one frame per timestep."""
    return [np.concatenate(per_t, axis=1) for per_t in zip(*frames_per_agent)]


def get_eval_frames(env_name, inner_env, ep_states, *, ep_obs=None, ep_actions=None):
    """Render `ep_states` to a list of RGB frames using the env's own renderer."""
    if env_name in ("lbf", "lbf-reward-shaping"):
        from marl.eval_lbf import _render_lbf_eval_frames
        return _render_lbf_eval_frames(inner_env, ep_states)
    if env_name == "card-game":
        from envs.card_game.rendering import render_card_game_eval_frames
        return render_card_game_eval_frames(
            ep_states, ep_obs=ep_obs, ep_actions=ep_actions,
        )
    if env_name == "hanabi":
        from envs.hanabi.rendering import render_hanabi_eval_frames
        return render_hanabi_eval_frames(ep_states)
    if env_name == "dual-destination":
        from envs.dual_destination.rendering import render_dual_destination_ego_frames
        f0, f1 = render_dual_destination_ego_frames(inner_env, ep_states)
        return _composite_per_agent([f0, f1])
    if env_name == "multi-destination-spread":
        from envs.multi_destination_spread.rendering import (
            render_multi_destination_ego_frames,
        )
        return _composite_per_agent(
            render_multi_destination_ego_frames(inner_env, ep_states)
        )
    raise NotImplementedError(
        f"No eval-frame renderer registered for env '{env_name}'. "
        "(overcooked-v1 is handled inline in common.eval_media.render_and_log_video.)"
    )
