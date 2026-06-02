"""Eval rollout and frame rendering for the Multi-Destination Spread env.

`rollout_states` runs one greedy, parameter-shared episode over all agents (the
2-agent `evaluation.vis_episodes.run_episode_with_states` does not generalise to
four agents), and `render_multi_destination_ego_frames` turns the resulting states
into per-agent ego-view frames for the per-checkpoint gifs.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def rollout_states(rng, inner_env, params, policy, max_steps: int) -> list:
    """Run one greedy episode with `policy` (shared params) driving every agent.

    Returns the list of `WrappedEnvState` visited (length `max_steps + 1`, including
    the reset state), matching what the renderers below expect.
    """
    n = inner_env.num_agents
    agents = inner_env.agents

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = inner_env.reset(reset_rng)
    hstate = policy.init_hstate(n)
    done = jnp.zeros((n,), dtype=bool)

    states = [env_state]
    for _ in range(max_steps):
        obs_batch = jnp.stack([obs[a] for a in agents], axis=0)        # (n, obs_dim)
        avail = inner_env.get_avail_actions(env_state)
        avail_batch = jnp.stack([avail[a] for a in agents], axis=0)    # (n, A)

        rng, act_rng, step_rng = jax.random.split(rng, 3)
        action, hstate = policy.get_action(
            params,
            obs_batch.reshape(1, n, -1),
            done.reshape(1, n),
            avail_batch.reshape(1, n, -1),
            hstate, act_rng, greedy=True,
        )
        action = action.reshape(n)
        env_act = {a: action[i] for i, a in enumerate(agents)}

        obs, env_state, _, dones, _ = inner_env.step(step_rng, env_state, env_act)
        states.append(env_state)
        done = jnp.array([dones[a] for a in agents])
        if bool(dones["__all__"]):
            break
    return states


def render_multi_destination_ego_frames(
    inner_env, ep_states, upscale: int = 8
) -> list[list[np.ndarray]]:
    """Render per-agent ego-view frames from a rollout.

    `ep_states` is the list of `WrappedEnvState` (so `ws.env_state` is the
    `MultiDestinationSpreadState`). Returns one frame list per agent, each a list
    of uint8 `(H, W, 3)` arrays upscaled by `upscale`.
    """
    frames: list[list[np.ndarray]] = [[] for _ in range(inner_env.num_agents)]
    for ws in ep_states:
        state = ws.env_state
        for i in range(inner_env.num_agents):
            img = np.asarray(inner_env._render_agent_view(state, i)).astype(np.uint8)
            if upscale > 1:
                img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)
            frames[i].append(img)
    return frames


def save_observation_montage(env, out_path: str, n_preview_steps: int = 3) -> str:
    """Render every agent's ego-view obs over a few steps and save a labelled montage PNG.

    Used by the observation test and for eyeballing a layout before training.
    `matplotlib`, `jax`, and `os` are imported lazily so the (training-time) gif
    path above stays free of the plotting dependency.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs, state = env.reset(jax.random.PRNGKey(0))
    del obs
    # Spread the four agents toward their nearest cardinal goals for a livelier
    # preview (N/W/E/S); clipped by the env to whatever the layout allows.
    spread = {"agent_0": 0, "agent_1": 6, "agent_2": 2, "agent_3": 4}  # N, W, E, S
    states = [state]
    s = state
    for t in range(n_preview_steps):
        _, s, _, _, _ = env.step(
            jax.random.PRNGKey(t + 1), s,
            {a: jnp.int32(spread[a]) for a in env.agents},
        )
        states.append(s)

    rows = len(states)
    cols = env.num_agents
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.4 * rows), squeeze=False)
    for t, st in enumerate(states):
        for idx in range(cols):
            view = np.asarray(env._render_agent_view(st.env_state, idx)).astype(np.uint8)
            ax = axes[t][idx]
            ax.imshow(view, interpolation="nearest")
            ax.set_title(f"t={t}  agent_{idx}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Multi-Destination Spread observations\n"
        "ego=RED  partners=BLUE  goals=GREEN  walls=GREY  bg=white",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
