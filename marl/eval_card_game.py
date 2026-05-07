"""Card-game-specific eval visualization: attention grids and multi-episode videos."""
import os

import jax
import numpy as np
from PIL import Image
from moviepy import ImageSequenceClip

from evaluation.vis_episodes import run_episode_with_states, _overlay_attention
from marl.eval_utils import (
    _draw_choice_on_cell,
    _draw_message_on_cell,
)


def _log_card_game_attention_grid(frames, attn_data, ep_actions, tag, video_dir, logger,
                                  ep_messages=None, card_permutation=None):
    """Log a 2xT grid image: row 0 = agent 0 attention, row 1 = agent 1 attention.

    Each cell shows the scene with the attention heatmap overlaid.
    The last column shows the agents' card choices as thick colored borders
    drawn on top of the spot marker.
    """
    import wandb

    maps_0 = attn_data.get("agent_0", [])
    maps_1 = attn_data.get("agent_1", [])
    if not maps_0 or not maps_1:
        print("[card_game] Missing attention maps, skipping grid.")
        return

    from envs.card_game.rendering import render_card_game_minimal

    # `frames` is kept in the signature for backward compat but no longer
    # used as a backdrop — we render fresh per timestep so the embedded
    # pixel-art timestep counter updates each cell.
    n_steps = min(len(maps_0), len(maps_1))
    if frames is not None:
        n_steps = min(n_steps, len(frames) - 1)
    scale = 32

    agent0_color = np.array([255, 140, 0], dtype=np.uint8)    # orange
    agent1_color = np.array([255, 0, 255], dtype=np.uint8)    # magenta

    last_action = ep_actions[-1] if ep_actions else (-1, -1)

    def _draw_choice(cell, choice, agent_idx):
        if card_permutation is not None:
            matches = np.where(card_permutation == choice)[0]
            if len(matches) > 0:
                _draw_choice_on_cell(cell, int(matches[0]), agent_idx, scale)
        else:
            _draw_choice_on_cell(cell, choice, agent_idx, scale)

    row_0 = []  # agent 0 attention (Oranges)
    row_1 = []  # agent 1 attention (RdPu)
    for t in range(n_steps):
        base_img = render_card_game_minimal(card_permutation, t + 1)
        base_np = np.array(base_img)
        base_up = np.array(Image.fromarray(base_np).resize(
            (base_np.shape[1] * scale, base_np.shape[0] * scale), Image.NEAREST,
        ))

        cell_0 = _overlay_attention(base_up, maps_0[t], "Oranges", alpha=0.6).copy()
        cell_1 = _overlay_attention(base_up, maps_1[t], "RdPu", alpha=0.6).copy()

        # Messages sent at step t-1 become visible in observation t.
        if ep_messages and (t - 1) >= 0 and (t - 1) < len(ep_messages):
            _draw_message_on_cell(
                cell_0, ep_messages[t - 1][1], scale,
                color=agent1_color, card_permutation=card_permutation,
            )
            _draw_message_on_cell(
                cell_1, ep_messages[t - 1][0], scale,
                color=agent0_color, card_permutation=card_permutation,
            )

        if t == n_steps - 1 and last_action[0] >= 0:
            _draw_choice(cell_0, last_action[0], 0)
        if t == n_steps - 1 and last_action[1] >= 0:
            _draw_choice(cell_1, last_action[1], 1)

        row_0.append(cell_0)
        row_1.append(cell_1)

    # Concatenate: each row is T frames side by side, then stack 2 rows
    padding = 4
    pad_color = np.array([255, 255, 255], dtype=np.uint8)
    cell_h, cell_w = row_0[0].shape[:2]

    grid_w = n_steps * cell_w + (n_steps - 1) * padding
    grid_h = 2 * cell_h + padding
    grid = np.full((grid_h, grid_w, 3), pad_color, dtype=np.uint8)

    for t in range(n_steps):
        x = t * (cell_w + padding)
        grid[0:cell_h, x:x + cell_w] = row_0[t]
        grid[cell_h + padding:grid_h, x:x + cell_w] = row_1[t]

    # Save locally and log to wandb
    os.makedirs(video_dir, exist_ok=True)
    grid_path = f"{video_dir}/attention_grid.png"
    Image.fromarray(grid).save(grid_path)
    print(f"[card_game] Saved attention grid: {grid_path} ({grid_w}x{grid_h} px)")

    logger.log({f"{tag}/attention_grid": wandb.Image(grid_path)}, commit=False)


def _log_card_game_action_distributions(
    inner_env, policy, params, max_steps, tag, logger,
    feed_attn_dims=None, ja_card_masks=None,
    num_episodes=30, rng_seed_base=300,
):
    """Print + log per-agent action distributions in EACH AGENT'S OWN VIEW
    frame across `num_episodes` SP eps.

    For each pick (decision step) and each message (deliberation step), the
    agent's GT-frame action is mapped to two view-frame bins:
      - view_pos:   column 0..NUM_CARDS-1 in the agent's OP-shuffled view
      - view_color: appearance colour idx after the agent's OP recolouring

    Concentration in either bin = the agent has converged on a positional or
    colour convention. With OP off both view-frame and gt-frame coincide.
    """
    from envs.card_game.rendering import NUM_CARDS

    def _walk(state, attr):
        s = state
        while s is not None:
            if hasattr(s, attr):
                return s
            s = getattr(s, "env_state", None)
        return None

    counts = {
        ("picks", "view_pos"): np.zeros((2, NUM_CARDS), dtype=np.int64),
        ("picks", "view_color"): np.zeros((2, NUM_CARDS), dtype=np.int64),
        ("messages", "view_pos"): np.zeros((2, NUM_CARDS), dtype=np.int64),
        ("messages", "view_color"): np.zeros((2, NUM_CARDS), dtype=np.int64),
    }

    for ep in range(num_episodes):
        ep_rng = jax.random.PRNGKey(rng_seed_base + ep)
        ep_states, ep_actions, ep_messages = run_episode_with_states(
            ep_rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=False,
            feed_other_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
        )
        n_steps = min(len(ep_messages), len(ep_actions))
        for t in range(n_steps):
            is_decision = (t == n_steps - 1)
            # On the decision step the env auto-resets, so use the prior
            # state's (still-current) per-agent transforms.
            action_state = ep_states[t - 1] if (is_decision and t > 0) else ep_states[t]
            base_state = _walk(action_state, "card_permutation")
            card_perm = (
                np.asarray(base_state.card_permutation) if base_state is not None
                else np.arange(NUM_CARDS, dtype=np.int32)
            )
            perm_state = _walk(action_state, "per_agent_perm")
            recol_state = _walk(action_state, "per_agent_recolouring")
            for ai in (0, 1):
                gt = int(ep_actions[t][ai]) if is_decision else int(ep_messages[t][ai])
                if gt < 0:
                    continue
                if perm_state is not None:
                    pp = np.asarray(perm_state.per_agent_perm[f"agent_{ai}"])
                    matches = np.where(pp == gt)[0]
                else:
                    matches = np.where(card_perm == gt)[0]
                if len(matches) == 0:
                    continue
                view_pos = int(matches[0])
                if recol_state is not None:
                    recol = np.asarray(
                        recol_state.per_agent_recolouring[f"agent_{ai}"]
                    )
                    view_color = int(recol[gt])
                else:
                    view_color = gt
                phase = "picks" if is_decision else "messages"
                counts[(phase, "view_pos")][ai, view_pos] += 1
                counts[(phase, "view_color")][ai, view_color] += 1

    print(f"\n=== {tag} action distributions ({num_episodes} SP eps, view-frame) ===")
    for phase in ("picks", "messages"):
        print(f"  {phase}:")
        for ai in (0, 1):
            for label in ("view_pos", "view_color"):
                arr = counts[(phase, label)][ai]
                total = int(arr.sum())
                if total == 0:
                    continue
                pct = arr / total * 100.0
                counts_str = " ".join(f"{c:>5d}" for c in arr)
                pct_str = " ".join(f"{p:>5.1f}%" for p in pct)
                print(
                    f"    A{ai} {label:11s}: counts [{counts_str}]  "
                    f"pct [{pct_str}]"
                )
                for c in range(NUM_CARDS):
                    logger.log(
                        {f"{tag}/action_dist/{phase}/{label}/A{ai}_{c}_pct": float(pct[c])},
                        commit=False,
                    )
    print()


def _log_card_game_eval_video(inner_env, policy, params, max_steps, tag, video_dir, logger,
                               feed_attn_dims=None,
                               ja_card_masks=None,
                               num_episodes=30, fps=3):
    """Run multiple card game episodes and save a video with attention spots and choices."""
    import wandb
    from envs.card_game.rendering import render_card_game_minimal, _unwrap_card_game_state

    scale = 20
    padding = 4
    all_video_frames = []

    for ep in range(num_episodes):
        ep_rng = jax.random.PRNGKey(100 + ep)
        ep_states, attn_data, ep_actions, ep_messages = run_episode_with_states(
            ep_rng, inner_env, params, policy,
            params, policy, max_steps,
            collect_attention=True,
            feed_other_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
        )

        maps_0 = attn_data.get("agent_0", [])
        maps_1 = attn_data.get("agent_1", [])
        if not maps_0 or not maps_1:
            continue

        n_steps = min(len(maps_0), len(maps_1))
        es0 = _unwrap_card_game_state(ep_states[0])
        card_permutation = np.array(es0.card_permutation)
        last_action = ep_actions[-1] if ep_actions else (-1, -1)

        for t in range(n_steps):
            base_img = render_card_game_minimal(es0.card_permutation, t + 1)
            base_np = np.array(base_img)
            base_up = np.array(Image.fromarray(base_np).resize(
                (base_np.shape[1] * scale, base_np.shape[0] * scale), Image.NEAREST,
            ))

            cell_0 = _overlay_attention(base_up, maps_0[t], "Oranges", alpha=0.6).copy()
            cell_1 = _overlay_attention(base_up, maps_1[t], "RdPu", alpha=0.6).copy()

            if ep_messages and (t - 1) >= 0 and (t - 1) < len(ep_messages):
                a0_color = [255, 140, 0]    # orange
                a1_color = [255, 0, 255]    # magenta
                _draw_message_on_cell(
                    cell_0, ep_messages[t - 1][1], scale,
                    color=a1_color, card_permutation=card_permutation,
                )
                _draw_message_on_cell(
                    cell_1, ep_messages[t - 1][0], scale,
                    color=a0_color, card_permutation=card_permutation,
                )

            if t == n_steps - 1 and last_action[0] >= 0:
                choice_0 = last_action[0]
                choice_1 = last_action[1]
                matches_0 = np.where(card_permutation == choice_0)[0]
                if len(matches_0) > 0:
                    _draw_choice_on_cell(cell_0, int(matches_0[0]), 0, scale)
                matches_1 = np.where(card_permutation == choice_1)[0]
                if len(matches_1) > 0:
                    _draw_choice_on_cell(cell_1, int(matches_1[0]), 1, scale)

            cell_h, cell_w = cell_0.shape[:2]
            frame = np.full((2 * cell_h + padding, cell_w, 3), 255, dtype=np.uint8)
            frame[:cell_h] = cell_0
            frame[cell_h + padding:] = cell_1
            all_video_frames.append(frame)

    if not all_video_frames:
        print("[card_game] No frames for eval video")
        return

    os.makedirs(video_dir, exist_ok=True)
    video_path = f"{video_dir}/eval_card_game.mp4"
    clip = ImageSequenceClip(all_video_frames, fps=fps)
    clip.write_videofile(video_path, fps=fps, codec='libx264', audio=False,
                         bitrate='8000k', preset='slow')
    logger.log_video(f"{tag}/eval_video", video_path, commit=False)
    print(f"[card_game] Saved eval video: {video_path} ({len(all_video_frames)} frames, {len(all_video_frames)/fps:.0f}s)")


def _log_card_game_xp_videos(inner_env, policy, all_params, max_steps, tag, video_dir, logger,
                              feed_attn_dims=None, ja_card_masks=None,
                              seed_pairs=None,
                              num_episodes=10, fps=3):
    """Generate cross-play videos: pair seed_i (agent 0) with seed_j (agent 1).

    Records a few episodes for each off-diagonal pair and logs as wandb videos.
    `seed_pairs` is an optional list of (i, j) tuples; if None, every off-diagonal
    pair is rendered.
    """
    import wandb

    num_seeds = jax.tree.leaves(all_params)[0].shape[0]
    scale = 20
    padding = 4

    if seed_pairs is None:
        seed_pairs = [
            (i, j) for i in range(num_seeds) for j in range(i + 1, num_seeds)
        ]

    for seed_i, seed_j in seed_pairs:
            params_i = jax.tree.map(lambda x: x[seed_i], all_params)
            params_j = jax.tree.map(lambda x: x[seed_j], all_params)

            all_video_frames = []
            for ep in range(num_episodes):
                ep_rng = jax.random.PRNGKey(5000 + seed_i * 1000 + seed_j * 100 + ep)
                ep_states, attn_data, ep_actions, ep_messages = run_episode_with_states(
                    ep_rng, inner_env, params_i, policy,
                    params_j, policy, max_steps,
                    collect_attention=True,
                    feed_other_attn_dims=feed_attn_dims,
                    ja_card_masks=ja_card_masks,
                )

                maps_0 = attn_data.get("agent_0", [])
                maps_1 = attn_data.get("agent_1", [])
                if not maps_0 or not maps_1:
                    continue

                n_steps = min(len(maps_0), len(maps_1))
                from envs.card_game.rendering import (
                    _unwrap_card_game_state,
                    render_card_game_minimal,
                )
                es0 = _unwrap_card_game_state(ep_states[0])
                card_permutation = np.array(es0.card_permutation)
                last_action = ep_actions[-1] if ep_actions else (-1, -1)

                for t in range(n_steps):
                    base_img = render_card_game_minimal(es0.card_permutation, t + 1)
                    base_np = np.array(base_img)
                    base_up = np.array(Image.fromarray(base_np).resize(
                        (base_np.shape[1] * scale, base_np.shape[0] * scale), Image.NEAREST,
                    ))

                    cell_0 = _overlay_attention(base_up, maps_0[t], "Oranges", alpha=0.6).copy()
                    cell_1 = _overlay_attention(base_up, maps_1[t], "RdPu", alpha=0.6).copy()

                    if ep_messages and (t - 1) >= 0 and (t - 1) < len(ep_messages):
                        a0_color = [255, 140, 0]
                        a1_color = [255, 0, 255]
                        _draw_message_on_cell(
                            cell_0, ep_messages[t - 1][1], scale,
                            color=a1_color, card_permutation=card_permutation,
                        )
                        _draw_message_on_cell(
                            cell_1, ep_messages[t - 1][0], scale,
                            color=a0_color, card_permutation=card_permutation,
                        )

                    if t == n_steps - 1 and last_action[0] >= 0:
                        choice_0 = last_action[0]
                        choice_1 = last_action[1]
                        matches_0 = np.where(card_permutation == choice_0)[0]
                        if len(matches_0) > 0:
                            _draw_choice_on_cell(cell_0, int(matches_0[0]), 0, scale)
                        matches_1 = np.where(card_permutation == choice_1)[0]
                        if len(matches_1) > 0:
                            _draw_choice_on_cell(cell_1, int(matches_1[0]), 1, scale)

                    cell_h, cell_w = cell_0.shape[:2]
                    frame = np.full((2 * cell_h + padding, cell_w, 3), 255, dtype=np.uint8)
                    frame[:cell_h] = cell_0
                    frame[cell_h + padding:] = cell_1
                    all_video_frames.append(frame)

            if not all_video_frames:
                continue

            os.makedirs(video_dir, exist_ok=True)
            video_path = f"{video_dir}/xp_seed{seed_i}_vs_seed{seed_j}.mp4"
            clip = ImageSequenceClip(all_video_frames, fps=fps)
            clip.write_videofile(video_path, fps=fps, codec='libx264', audio=False,
                                 bitrate='8000k', preset='slow')
            logger.log_video(f"{tag}/xp_video_s{seed_i}_vs_s{seed_j}", video_path, commit=False)
            print(f"[card_game] XP video s{seed_i} vs s{seed_j}: {video_path} ({len(all_video_frames)} frames)")


def _gt_to_view_col(state, agent_idx: int, gt_card_id: int, card_perm: np.ndarray):
    """Map a GT card identity to the agent's view column under the OP wrappers.

    Returns None when the pick is invalid or no position-shuffle wrapper is
    present (e.g. running on a config without OP, where view = physical).
    """
    if gt_card_id is None or gt_card_id < 0:
        return None
    phys_cols = np.where(card_perm == int(gt_card_id))[0]
    if len(phys_cols) == 0:
        return None
    phys_col = int(phys_cols[0])
    s = state
    while s is not None and not hasattr(s, "per_agent_perm"):
        s = getattr(s, "env_state", None)
    if s is None:
        return phys_col
    pos_perm = np.asarray(s.per_agent_perm[f"agent_{agent_idx}"])
    view_cols = np.where(pos_perm == phys_col)[0]
    return int(view_cols[0]) if len(view_cols) else None


def _log_card_game_per_agent_obs_video(
    inner_env, policy, params, max_steps, tag, video_dir, logger,
    feed_attn_dims=None, ja_card_masks=None,
    num_episodes=30, fps=3,
    params_partner=None, video_filename="eval_card_game_per_agent.mp4",
    rng_seed_base=100, video_log_key=None,
):
    """Eval video built from each agent's actual observation.

    Under Other-Play each agent sees a different shuffle and recolouring, so
    overlaying attention on the canonical scene is misleading. This function
    reshapes the policy's input obs back to an image (it already contains the
    OP-transformed cards, the ego border, the partner-message dot and the
    decision indicator) and overlays the agent's own attention map on top.

    The only decoration drawn here is the per-agent picked-card border on the
    decision step and a timestep label.

    `params_partner` (optional) lets the caller pair `params` (agent 0) with a
    different partner (agent 1) for cross-play renderings; defaults to `params`
    for self-play. `video_filename`, `rng_seed_base`, `video_log_key` let the
    XP-pair caller distinguish per-pair videos and log keys.
    """
    from envs.card_game.rendering import (
        TILE_PIXELS, GRID_ROWS, GRID_COLS, _unwrap_card_game_state,
    )

    img_h = GRID_ROWS * TILE_PIXELS
    img_w = GRID_COLS * TILE_PIXELS
    scale = 20
    padding = 4
    all_video_frames: list = []

    if params_partner is None:
        params_partner = params

    for ep in range(num_episodes):
        ep_rng = jax.random.PRNGKey(rng_seed_base + ep)
        ep_states, attn_data, ep_actions, ep_messages, ep_obs = run_episode_with_states(
            ep_rng, inner_env, params, policy, params_partner, policy, max_steps,
            collect_attention=True,
            collect_obs=True,
            feed_other_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
        )

        maps_0 = attn_data.get("agent_0", [])
        maps_1 = attn_data.get("agent_1", [])
        if not maps_0 or not maps_1 or not ep_obs:
            continue

        n_steps = min(len(maps_0), len(maps_1), len(ep_obs))
        es0 = _unwrap_card_game_state(ep_states[0])
        card_perm = np.asarray(es0.card_permutation)
        last_action = ep_actions[-1] if ep_actions else (-1, -1)

        for t in range(n_steps):
            obs_t = ep_obs[t]
            base_0 = (
                np.asarray(obs_t["agent_0"]).reshape(img_h, img_w, 3) * 255
            ).astype(np.uint8)
            base_1 = (
                np.asarray(obs_t["agent_1"]).reshape(img_h, img_w, 3) * 255
            ).astype(np.uint8)
            up_0 = np.array(Image.fromarray(base_0).resize(
                (img_w * scale, img_h * scale), Image.NEAREST,
            ))
            up_1 = np.array(Image.fromarray(base_1).resize(
                (img_w * scale, img_h * scale), Image.NEAREST,
            ))

            cell_0 = _overlay_attention(up_0, maps_0[t], "Oranges", alpha=0.6).copy()
            cell_1 = _overlay_attention(up_1, maps_1[t], "RdPu", alpha=0.6).copy()

            is_decision = (t == n_steps - 1)
            if is_decision:
                pick_0_view = _gt_to_view_col(
                    ep_states[t], 0, int(last_action[0]), card_perm,
                )
                pick_1_view = _gt_to_view_col(
                    ep_states[t], 1, int(last_action[1]), card_perm,
                )
                # Env scores on GT-frame match (card_game.py:268). Highlight
                # solved episodes in yellow so it's obvious at a glance.
                solved = (
                    int(last_action[0]) >= 0
                    and int(last_action[1]) >= 0
                    and int(last_action[0]) == int(last_action[1])
                )
                box_color = [255, 255, 0] if solved else None
                if pick_0_view is not None:
                    _draw_choice_on_cell(cell_0, pick_0_view, 0, scale, color=box_color)
                if pick_1_view is not None:
                    _draw_choice_on_cell(cell_1, pick_1_view, 1, scale, color=box_color)

            cell_h, cell_w = cell_0.shape[:2]
            frame = np.full(
                (2 * cell_h + padding, cell_w, 3), 255, dtype=np.uint8,
            )
            frame[:cell_h] = cell_0
            frame[cell_h + padding:] = cell_1
            all_video_frames.append(frame)

    if not all_video_frames:
        print("[card_game] No frames for per-agent obs eval video")
        return

    os.makedirs(video_dir, exist_ok=True)
    video_path = f"{video_dir}/{video_filename}"
    clip = ImageSequenceClip(all_video_frames, fps=fps)
    clip.write_videofile(
        video_path, fps=fps, codec='libx264', audio=False,
        bitrate='8000k', preset='slow',
    )
    log_key = video_log_key or f"{tag}/eval_video_per_agent"
    logger.log_video(log_key, video_path, commit=False)
    print(
        f"[card_game] Saved per-agent eval video: {video_path} "
        f"({len(all_video_frames)} frames, {len(all_video_frames)/fps:.0f}s)"
    )


def _log_card_game_per_agent_xp_videos(
    inner_env, policy, all_params, max_steps, tag, video_dir, logger,
    feed_attn_dims=None, ja_card_masks=None,
    seed_pairs=None, num_episodes=5, fps=3,
):
    """Per-agent obs cross-play videos: one mp4 per (i, j) pair, agent 0 = seed_i.

    Mirrors `_log_card_game_xp_videos` (canonical-frame XP) but uses the
    per-agent OP-recoloured/shuffled view from `_log_card_game_per_agent_obs_video`
    so each agent's actual policy input is what's shown.
    """
    num_seeds = jax.tree.leaves(all_params)[0].shape[0]
    if seed_pairs is None:
        seed_pairs = [
            (i, j) for i in range(num_seeds) for j in range(i + 1, num_seeds)
        ]
    print(f"[card_game] per-agent XP videos: {len(seed_pairs)} pair(s) -> {video_dir}")

    for seed_i, seed_j in seed_pairs:
        params_i = jax.tree.map(lambda x, _i=seed_i: x[_i], all_params)
        params_j = jax.tree.map(lambda x, _j=seed_j: x[_j], all_params)
        _log_card_game_per_agent_obs_video(
            inner_env, policy, params_i, max_steps,
            tag=tag, video_dir=video_dir, logger=logger,
            feed_attn_dims=feed_attn_dims, ja_card_masks=ja_card_masks,
            num_episodes=num_episodes, fps=fps,
            params_partner=params_j,
            video_filename=f"per_agent_xp_s{seed_i}_vs_s{seed_j}.mp4",
            rng_seed_base=5000 + seed_i * 1000 + seed_j * 100,
            video_log_key=f"{tag}/per_agent_xp_s{seed_i}_vs_s{seed_j}",
        )
