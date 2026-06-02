"""Shared eval rollout + video logging used by all trainers.

This is the one place that turns a trained policy into a logged episode video,
for both per-checkpoint and end-of-training videos, across every env. Env-specific
frame rendering is delegated to `envs.render_registry.get_eval_frames`; Overcooked
(whose visualizer writes the mp4 inline) is special-cased here.

`rollout_states` is a generic, parameter-shared N-agent rollout — the 2-agent
`evaluation.vis_episodes.run_episode_with_states` does not generalise to more
agents. For JA envs that need attention/obs/action overlays, callers can pass
pre-collected `ep_states`/`ep_obs`/`ep_actions` from `run_episode_with_states`
instead of using `rollout_states`.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from envs.render_registry import get_eval_frames


def rollout_states(
    rng,
    inner_env,
    params,
    policy,
    max_steps,
    *,
    collect_obs_actions=False,
    greedy=True,
):
    """Run one parameter-shared episode driving every agent with `policy`.

    Returns the list of `WrappedEnvState` visited (incl. the reset state). With
    `collect_obs_actions=True`, also returns per-step obs dicts and action tuples
    (needed by renderers like the card game that highlight picks).
    """
    n = inner_env.num_agents
    agents = inner_env.agents

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = inner_env.reset(reset_rng)
    hstate = policy.init_hstate(n)
    done = jnp.zeros((n,), dtype=bool)

    states = [env_state]
    obs_list: list = []
    act_list: list = []
    for _ in range(max_steps):
        obs_batch = jnp.stack([obs[a] for a in agents], axis=0)
        avail = inner_env.get_avail_actions(env_state)
        avail_batch = jnp.stack([avail[a] for a in agents], axis=0)

        rng, act_rng, step_rng = jax.random.split(rng, 3)
        action, hstate = policy.get_action(
            params,
            obs_batch.reshape(1, n, -1),
            done.reshape(1, n),
            avail_batch.reshape(1, n, -1),
            hstate, act_rng, greedy=greedy,
        )
        action = action.reshape(n)
        if collect_obs_actions:
            obs_list.append({a: np.asarray(obs[a]) for a in agents})
            act_list.append(tuple(int(action[i]) for i in range(n)))

        env_act = {a: action[i] for i, a in enumerate(agents)}
        obs, env_state, _, dones, _ = inner_env.step(step_rng, env_state, env_act)
        states.append(env_state)
        done = jnp.array([dones[a] for a in agents])
        if bool(dones["__all__"]):
            break

    if collect_obs_actions:
        return states, obs_list, act_list
    return states


def render_and_log_video(inner_env, env_name, ep_states, tag, savedir, logger, *,
                         ep_obs=None, ep_actions=None, fps=10):
    """Render `ep_states` to an mp4 (via the env-render registry) and log it to W&B.

    Overcooked is special-cased: its visualizer writes the mp4 directly.
    Returns the written video path.
    """
    os.makedirs(savedir, exist_ok=True)
    video_path = os.path.join(savedir, tag.replace("/", "_") + ".mp4")

    if env_name == "overcooked-v1":
        from envs.overcooked.adhoc_overcooked_visualizer import AdHocOvercookedVisualizer
        AdHocOvercookedVisualizer().animate_mp4(
            [s.env_state for s in ep_states], inner_env.agent_view_size,
            filename=video_path, pixels_per_tile=32, fps=fps,
        )
    else:
        frames = get_eval_frames(
            env_name, inner_env, ep_states, ep_obs=ep_obs, ep_actions=ep_actions,
        )
        from moviepy import ImageSequenceClip
        ImageSequenceClip(list(frames), fps=fps).write_videofile(
            video_path, fps=fps, codec="libx264", audio=False,
        )

    if logger is not None and getattr(logger, "run", None) is not None:
        logger.log_video(tag, video_path, commit=False)
    return video_path


def _default_video_greedy(env_name: str) -> bool:
    """Default action-selection mode for qualitative eval videos."""
    # Multi-destination spread starts all parameter-shared agents on the same
    # cell with identical observations. Greedy eval makes them all choose the
    # same action, collide, and stay static forever; sampled eval breaks that
    # symmetry like training rollouts do.
    return env_name != "multi-destination-spread"


def rollout_and_log_video(rng, inner_env, env_name, params, policy, max_steps,
                          tag, savedir, logger, *, fps=10, greedy=None):
    """Convenience: generic N-agent rollout + render + log, for non-JA trainers.

    Collects obs/actions only for envs whose renderer needs them (card game).
    """
    if greedy is None:
        greedy = _default_video_greedy(env_name)
    needs_obs_actions = env_name == "card-game"
    roll = rollout_states(
        rng, inner_env, params, policy, max_steps,
        collect_obs_actions=needs_obs_actions,
        greedy=greedy,
    )
    if needs_obs_actions:
        ep_states, ep_obs, ep_actions = roll
    else:
        ep_states, ep_obs, ep_actions = roll, None, None
    return render_and_log_video(
        inner_env, env_name, ep_states, tag, savedir, logger,
        ep_obs=ep_obs, ep_actions=ep_actions, fps=fps,
    )
