"""Eval-frame rendering for the Dual Destination env.

Renders the ego-centric agent views (ego red, partner blue, goals green, walls
grey) from a rollout's states, for the DreamTeam-style per-checkpoint gifs.
"""
from __future__ import annotations

import numpy as np


def render_dual_destination_ego_frames(
    inner_env, ep_states, upscale: int = 8
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render per-agent ego-view frames from a rollout.

    `ep_states` is the list of `WrappedEnvState` returned by
    `run_episode_with_states` (so `ws.env_state` is the `DualDestinationState`).
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


def save_observation_montage(env, out_path: str, n_preview_steps: int = 3) -> str:
    """Render both agents' ego-view observations over a few steps and save a
    labelled montage PNG (returns the path).

    Used by the observation test and for eyeballing a layout before training.
    `matplotlib`, `jax`, and `os` are imported lazily so the (training-time) gif
    path above stays free of the plotting dependency.
    """
    import os

    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs, state = env.reset(jax.random.PRNGKey(0))
    del obs
    # Non-crossing nudges toward the two goals (up/right vs down/left) for a
    # livelier preview; clipped by the env to whatever the layout allows.
    a0, a1 = [3, 3, 0], [1, 1, 2]
    states = [state]
    s = state
    for t in range(min(n_preview_steps, 3)):
        _, s, _, _, _ = env.step(
            jax.random.PRNGKey(t + 1), s,
            {"agent_0": jnp.int32(a0[t]), "agent_1": jnp.int32(a1[t])},
        )
        states.append(s)

    rows = len(states)
    fig, axes = plt.subplots(rows, 2, figsize=(5, 2.4 * rows), squeeze=False)
    for t, st in enumerate(states):
        for col, idx in enumerate((0, 1)):
            view = np.asarray(env._render_agent_view(st.env_state, idx)).astype(np.uint8)
            ax = axes[t][col]
            ax.imshow(view, interpolation="nearest")
            ax.set_title(f"t={t}  agent_{idx} observation", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Dual Destination observations\n"
        "ego=RED  partner=BLUE  goals=GREEN  walls=GREY  bg=white",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
