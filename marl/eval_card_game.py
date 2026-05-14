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
                                  ep_messages=None, card_permutation=None,
                                  ep_obs=None, ep_states=None):
    """Log a 2xT grid image: row 0 = agent 0 attention, row 1 = agent 1 attention.

    When `ep_obs` is provided, each cell uses the *agent's own* obs as the
    backdrop — under OP this shows the shuffled+recoloured view that the
    policy actually saw, so the attention overlay aligns with the agent's
    perspective. The last column shows white borders around picked cards.
    """
    import wandb

    maps_0 = attn_data.get("agent_0", [])
    maps_1 = attn_data.get("agent_1", [])
    if not maps_0 or not maps_1:
        print("[card_game] Missing attention maps, skipping grid.")
        return

    from envs.card_game.rendering import (
        render_card_game_minimal,
        _stamp_label_np,
        _A0_PATTERN_SMALL,
        _A1_PATTERN_SMALL,
        _gt_pick_to_view_col,
        _recolour_message_dot,
        GRID_ROWS,
        GRID_COLS,
        TILE_PIXELS,
    )

    n_steps = min(len(maps_0), len(maps_1))
    if frames is not None:
        n_steps = min(n_steps, len(frames) - 1)
    scale = 32
    h_px = GRID_ROWS * TILE_PIXELS
    w_px = GRID_COLS * TILE_PIXELS

    agent0_color = np.array([255, 140, 0], dtype=np.uint8)
    agent1_color = np.array([255, 0, 255], dtype=np.uint8)
    last_action = ep_actions[-1] if ep_actions else (-1, -1)
    won = (
        int(last_action[0]) >= 0 and int(last_action[1]) >= 0
        and int(last_action[0]) == int(last_action[1])
    )
    decision_border = [255, 255, 0] if won else [255, 255, 255]

    row_0 = []
    row_1 = []
    for t in range(n_steps):
        if ep_obs is not None and t < len(ep_obs):
            base_0 = (np.asarray(ep_obs[t]["agent_0"]).reshape(h_px, w_px, 3) * 255).astype(np.uint8)
            base_1 = (np.asarray(ep_obs[t]["agent_1"]).reshape(h_px, w_px, 3) * 255).astype(np.uint8)
            if ep_states is not None and t < len(ep_states):
                _recolour_message_dot(base_0, ep_states[t], 0)
                _recolour_message_dot(base_1, ep_states[t], 1)
            up_0 = np.array(Image.fromarray(base_0).resize(
                (w_px * scale, h_px * scale), Image.NEAREST,
            ))
            up_1 = np.array(Image.fromarray(base_1).resize(
                (w_px * scale, h_px * scale), Image.NEAREST,
            ))
        else:
            base_img = render_card_game_minimal(card_permutation, t + 1)
            base_np = np.array(base_img)
            up = np.array(Image.fromarray(base_np).resize(
                (w_px * scale, h_px * scale), Image.NEAREST,
            ))
            up_0 = up_1 = up
            if ep_messages and (t - 1) >= 0 and (t - 1) < len(ep_messages):
                # Draw colour-coded message dots on the canonical scene only;
                # under per-agent obs the obs already contains the dot.
                _draw_message_on_cell(
                    up_0, ep_messages[t - 1][1], scale,
                    color=agent1_color, card_permutation=card_permutation,
                )
                _draw_message_on_cell(
                    up_1, ep_messages[t - 1][0], scale,
                    color=agent0_color, card_permutation=card_permutation,
                )

        cell_0 = _overlay_attention(up_0, maps_0[t], "Oranges", alpha=0.6).copy()
        cell_1 = _overlay_attention(up_1, maps_1[t], "RdPu", alpha=0.6).copy()

        if t == n_steps - 1 and last_action[0] >= 0:
            if ep_obs is not None and ep_states is not None:
                state_for_perm = ep_states[t - 1] if t > 0 else ep_states[t]
                view_0 = _gt_pick_to_view_col(state_for_perm, 0, int(last_action[0]))
                view_1 = _gt_pick_to_view_col(state_for_perm, 1, int(last_action[1]))
            elif card_permutation is not None:
                m0 = np.where(card_permutation == int(last_action[0]))[0]
                m1 = np.where(card_permutation == int(last_action[1]))[0]
                view_0 = int(m0[0]) if len(m0) else -1
                view_1 = int(m1[0]) if len(m1) else -1
            else:
                view_0 = int(last_action[0])
                view_1 = int(last_action[1])
            if view_0 >= 0:
                _draw_choice_on_cell(cell_0, view_0, 0, scale, color=decision_border)
            if view_1 >= 0:
                _draw_choice_on_cell(cell_1, view_1, 1, scale, color=decision_border)

        _stamp_label_np(cell_0, _A0_PATTERN_SMALL, 1, 27, agent0_color, scale=scale)
        _stamp_label_np(cell_1, _A1_PATTERN_SMALL, 1, 27, agent1_color, scale=scale)

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


def _log_card_game_own_vs_partner_panel(
    inner_env, policy, params, max_steps, tag, video_dir, logger,
    feed_attn_dims=None, ja_card_masks=None,
    partner_feed_dim=5, num_episodes=10,
    rng_seed_base=8000,
    seed_idx=None,
):
    """For each agent, per episode: 2 rows × T cols.
       Top    = own head-averaged attention overlaid on own obs (agent's palette).
       Bottom = partner's attention translated to own view-slot frame, overlaid
                on own obs (partner's palette).
       Palette convention: agent 0 = "hot", agent 1 = "RdPu".

       This is the diagnostic for whether aux is working: the bottom row is
       what the agent receives as partner_feed; if attention has learned to
       decode it (aux_nll small) the top row should match.
    """
    import wandb
    from pathlib import Path
    from evaluation.card_game.attention_numbers import (
        _save_per_head_action_overlay,
    )

    if ja_card_masks is None:
        raise ValueError("own_vs_partner_panel requires ja_card_masks (set by JA_CARD_PARTNER_FEED).")

    os.makedirs(video_dir, exist_ok=True)
    img_h_const, img_w_const = 21, 35
    palettes = ("hot", "RdPu")  # by agent_idx

    for ep in range(num_episodes):
        ep_rng = jax.random.PRNGKey(rng_seed_base + ep * 17)
        ep_states, attn_data, ep_actions, ep_messages, ep_obs = run_episode_with_states(
            ep_rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=True,
            collect_obs=True,
            feed_other_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
            partner_feed_dim=partner_feed_dim,
        )

        T = len(ep_obs)
        if T == 0:
            continue
        is_decision_seq = [t == T - 1 for t in range(T)]
        maps0 = attn_data.get("agent_0", [])
        maps1 = attn_data.get("agent_1", [])
        if not maps0 or not maps1 or len(maps0) < T or len(maps1) < T:
            continue

        # Per-step head-averaged spatial attention, per agent.
        def _head_avg(a):
            a_sq = np.asarray(a).squeeze()
            if a_sq.ndim == 3:
                a_sq = a_sq.mean(axis=-1)
            return a_sq

        attn0 = np.stack([_head_avg(maps0[t]) for t in range(T)], axis=0)  # (T, fh, fw)
        attn1 = np.stack([_head_avg(maps1[t]) for t in range(T)], axis=0)

        card_masks_np = np.asarray(ja_card_masks)  # (5, fh, fw)

        def _per_card_mass(attn_t):
            # (5,) for each t — view-slot mass under the agent's own view.
            return np.einsum("thw,chw->tc", attn_t, card_masks_np)

        m0 = _per_card_mass(attn0)  # (T, 5) — agent 0 view-slot masses
        m1 = _per_card_mass(attn1)  # (T, 5)

        # Per-step perms — at step t both agents' perms come from state t.
        perms_0 = []
        perms_1 = []
        for t in range(T):
            s = ep_states[t]
            walk = s
            p0 = p1 = None
            while walk is not None:
                if hasattr(walk, "per_agent_perm"):
                    p0 = np.asarray(walk.per_agent_perm["agent_0"])
                    p1 = np.asarray(walk.per_agent_perm["agent_1"])
                    break
                walk = getattr(walk, "env_state", None)
            perms_0.append(p0 if p0 is not None else np.arange(5))
            perms_1.append(p1 if p1 is not None else np.arange(5))

        # Translate partner's per-card mass into ego view-slot frame.
        # phys[partner][canon_pos] = m_partner[k] when partner_perm[k] = canon_pos.
        # translated_for_ego[k] = phys_partner[ego_perm[k]] = partner's mass on the
        # canonical card sitting at ego's view-slot k.
        translated_for_0 = np.zeros((T, 5), dtype=np.float32)  # partner=1, ego=0
        translated_for_1 = np.zeros((T, 5), dtype=np.float32)
        for t in range(T):
            phys1 = np.zeros(5, dtype=np.float32)
            phys1[perms_1[t]] = m1[t]
            phys0 = np.zeros(5, dtype=np.float32)
            phys0[perms_0[t]] = m0[t]
            translated_for_0[t] = phys1[perms_0[t]]
            translated_for_1[t] = phys0[perms_1[t]]

        def _paint_card_mass(per_card_seq):
            # (T, 5) -> (T, fh, fw, 1): uniform fill per card region.
            painted = np.einsum("tc,chw->thw", per_card_seq, card_masks_np)
            return painted[..., None]

        for agent_idx, (agent_key, own_mass_attn, partner_mass) in enumerate([
            ("agent_0", attn0, translated_for_0),
            ("agent_1", attn1, translated_for_1),
        ]):
            obs_seq = []
            for o in ep_obs:
                a_obs = np.asarray(o[agent_key])
                img = a_obs[: img_h_const * img_w_const * 3].reshape(
                    img_h_const, img_w_const, 3
                )
                obs_seq.append(np.clip(img, 0.0, 1.0).astype(np.float32))
            obs_seq = np.stack(obs_seq, axis=0)

            own_row = own_mass_attn[..., None]                 # (T, fh, fw, 1) — raw spatial
            partner_row = _paint_card_mass(partner_mass)        # (T, fh, fw, 1) — synthetic per-card
            stacked = np.concatenate([own_row, partner_row], axis=-1)  # (T, fh, fw, 2)

            action_view_slots = [-1] * T
            if len(ep_actions) >= T and len(ep_states) >= T:
                state_dec = ep_states[T - 1]
                inv_recol = None
                pos_perm = None
                s = state_dec
                while s is not None:
                    if inv_recol is None and hasattr(s, "per_agent_inv_recolouring"):
                        inv_recol = np.asarray(s.per_agent_inv_recolouring[agent_key])
                    if pos_perm is None and hasattr(s, "per_agent_perm"):
                        pos_perm = np.asarray(s.per_agent_perm[agent_key])
                    s = getattr(s, "env_state", None)
                raw = int(ep_actions[T - 1][agent_idx])
                if raw >= 0 and inv_recol is not None and pos_perm is not None:
                    own_pick_gt = int(inv_recol[raw])
                    pos_perm_inv = np.argsort(pos_perm)
                    action_view_slots[T - 1] = int(pos_perm_inv[own_pick_gt])
            partner_msg_view_slots = [-1] * T

            own_palette = palettes[agent_idx]
            partner_palette = palettes[1 - agent_idx]

            out_path = Path(video_dir) / f"own_vs_partner_{agent_key}_ep{ep}.png"
            seed_label = "" if seed_idx is None else f"seed {seed_idx}  "
            human_title = (
                f"{seed_label}Agent {agent_idx}  episode {ep}  "
                f"(own attn vs partner attn translated to own frame)"
            )
            _save_per_head_action_overlay(
                per_head_seq=stacked,
                obs_seq=obs_seq,
                action_view_slots=action_view_slots,
                is_decision_seq=is_decision_seq,
                partner_msg_view_slots=partner_msg_view_slots,
                out_path=out_path,
                title=human_title,
                agent_idx=agent_idx,
                legend="  (top = own; bottom = partner translated)",
                row_labels=["own", "partner"],
                row_cmaps=[own_palette, partner_palette],
            )
            logger.log(
                {f"{tag}/own_vs_partner_{agent_key}_ep{ep}": wandb.Image(str(out_path))},
                commit=False,
            )
            print(f"[card_game] Saved own-vs-partner panel: {out_path}")


def _log_card_game_action_distributions(
    inner_env, policy, params, max_steps, tag, logger,
    feed_attn_dims=None, ja_card_masks=None,
    num_episodes=30, rng_seed_base=300,
    partner_feed_dim=5,
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
            partner_feed_dim=partner_feed_dim,
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
    print()


def _log_card_game_eval_video(inner_env, policy, params, max_steps, tag, video_dir, logger,
                               feed_attn_dims=None,
                               ja_card_masks=None,
                               num_episodes=30, fps=3,
                               partner_feed_dim=5):
    """Run multiple card game episodes and save a video with attention spots and choices."""
    import wandb
    from envs.card_game.rendering import (
        render_card_game_minimal,
        render_card_game_gt_frame,
        _unwrap_card_game_state,
        _stamp_label_np,
        _A0_PATTERN_SMALL,
        _A1_PATTERN_SMALL,
        _gt_pick_to_view_col,
        _recolour_message_dot,
        GRID_ROWS,
        GRID_COLS,
        TILE_PIXELS,
    )

    scale = 20
    padding = 4
    all_video_frames = []
    a0_color_np = np.array([255, 140, 0], dtype=np.uint8)
    a1_color_np = np.array([255, 0, 255], dtype=np.uint8)
    h_px = GRID_ROWS * TILE_PIXELS
    w_px = GRID_COLS * TILE_PIXELS

    for ep in range(num_episodes):
        ep_rng = jax.random.PRNGKey(100 + ep)
        ep_states, attn_data, ep_actions, ep_messages, ep_obs = run_episode_with_states(
            ep_rng, inner_env, params, policy,
            params, policy, max_steps,
            collect_attention=True,
            collect_obs=True,
            feed_other_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
            partner_feed_dim=partner_feed_dim,
        )

        maps_0 = attn_data.get("agent_0", [])
        maps_1 = attn_data.get("agent_1", [])
        if not maps_0 or not maps_1:
            continue

        n_steps = min(len(maps_0), len(maps_1))
        last_action = ep_actions[-1] if ep_actions else (-1, -1)

        for t in range(n_steps):
            # Per-agent obs as backdrop (under OP this shows shuffled +
            # recoloured view + agent's own message dot).
            base_0 = (np.asarray(ep_obs[t]["agent_0"]).reshape(h_px, w_px, 3) * 255).astype(np.uint8)
            base_1 = (np.asarray(ep_obs[t]["agent_1"]).reshape(h_px, w_px, 3) * 255).astype(np.uint8)
            _recolour_message_dot(base_0, ep_states[t], 0)
            _recolour_message_dot(base_1, ep_states[t], 1)
            up_0 = np.array(Image.fromarray(base_0).resize(
                (w_px * scale, h_px * scale), Image.NEAREST,
            ))
            up_1 = np.array(Image.fromarray(base_1).resize(
                (w_px * scale, h_px * scale), Image.NEAREST,
            ))

            cell_0 = _overlay_attention(up_0, maps_0[t], "Oranges", alpha=0.6).copy()
            cell_1 = _overlay_attention(up_1, maps_1[t], "RdPu", alpha=0.6).copy()

            if t == n_steps - 1 and last_action[0] >= 0:
                state_for_perm = ep_states[t - 1] if t > 0 else ep_states[t]
                view_0 = _gt_pick_to_view_col(state_for_perm, 0, int(last_action[0]))
                view_1 = _gt_pick_to_view_col(state_for_perm, 1, int(last_action[1]))
                won = int(last_action[0]) == int(last_action[1])
                decision_border = [255, 255, 0] if won else [255, 255, 255]
                if view_0 >= 0:
                    _draw_choice_on_cell(cell_0, view_0, 0, scale, color=decision_border)
                if view_1 >= 0:
                    _draw_choice_on_cell(cell_1, view_1, 1, scale, color=decision_border)

            _stamp_label_np(cell_0, _A0_PATTERN_SMALL, 1, 27, a0_color_np, scale=scale)
            _stamp_label_np(cell_1, _A1_PATTERN_SMALL, 1, 27, a1_color_np, scale=scale)

            is_decision_step = t == n_steps - 1
            cell_gt = render_card_game_gt_frame(
                ep_states[t],
                last_action if is_decision_step else None,
                is_decision_step,
                scale,
            )

            cell_h, cell_w = cell_0.shape[:2]
            frame = np.full((3 * cell_h + 2 * padding, cell_w, 3), 255, dtype=np.uint8)
            frame[:cell_h] = cell_0
            frame[cell_h + padding:2 * cell_h + padding] = cell_1
            frame[2 * cell_h + 2 * padding:] = cell_gt
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
                              num_episodes=10, fps=3,
                              partner_feed_dim=5):
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
                ep_states, attn_data, ep_actions, ep_messages, ep_obs = run_episode_with_states(
                    ep_rng, inner_env, params_i, policy,
                    params_j, policy, max_steps,
                    collect_attention=True,
                    collect_obs=True,
                    feed_other_attn_dims=feed_attn_dims,
                    ja_card_masks=ja_card_masks,
                    partner_feed_dim=partner_feed_dim,
                )

                maps_0 = attn_data.get("agent_0", [])
                maps_1 = attn_data.get("agent_1", [])
                if not maps_0 or not maps_1:
                    continue

                n_steps = min(len(maps_0), len(maps_1))
                from envs.card_game.rendering import (
                    _stamp_label_np,
                    _A0_PATTERN_SMALL,
                    _A1_PATTERN_SMALL,
                    _gt_pick_to_view_col,
                    _recolour_message_dot,
                    GRID_ROWS,
                    GRID_COLS,
                    TILE_PIXELS,
                )
                a0_color_np_xp = np.array([255, 140, 0], dtype=np.uint8)
                a1_color_np_xp = np.array([255, 0, 255], dtype=np.uint8)
                last_action = ep_actions[-1] if ep_actions else (-1, -1)
                h_px = GRID_ROWS * TILE_PIXELS
                w_px = GRID_COLS * TILE_PIXELS

                for t in range(n_steps):
                    base_0 = (np.asarray(ep_obs[t]["agent_0"]).reshape(h_px, w_px, 3) * 255).astype(np.uint8)
                    base_1 = (np.asarray(ep_obs[t]["agent_1"]).reshape(h_px, w_px, 3) * 255).astype(np.uint8)
                    _recolour_message_dot(base_0, ep_states[t], 0)
                    _recolour_message_dot(base_1, ep_states[t], 1)
                    up_0 = np.array(Image.fromarray(base_0).resize(
                        (w_px * scale, h_px * scale), Image.NEAREST,
                    ))
                    up_1 = np.array(Image.fromarray(base_1).resize(
                        (w_px * scale, h_px * scale), Image.NEAREST,
                    ))
                    cell_0 = _overlay_attention(up_0, maps_0[t], "Oranges", alpha=0.6).copy()
                    cell_1 = _overlay_attention(up_1, maps_1[t], "RdPu", alpha=0.6).copy()

                    if t == n_steps - 1 and last_action[0] >= 0:
                        state_for_perm = ep_states[t - 1] if t > 0 else ep_states[t]
                        view_0 = _gt_pick_to_view_col(state_for_perm, 0, int(last_action[0]))
                        view_1 = _gt_pick_to_view_col(state_for_perm, 1, int(last_action[1]))
                        won = int(last_action[0]) == int(last_action[1])
                        decision_border = [255, 255, 0] if won else [255, 255, 255]
                        if view_0 >= 0:
                            _draw_choice_on_cell(cell_0, view_0, 0, scale, color=decision_border)
                        if view_1 >= 0:
                            _draw_choice_on_cell(cell_1, view_1, 1, scale, color=decision_border)

                    _stamp_label_np(
                        cell_0, _A0_PATTERN_SMALL, 1, 27, a0_color_np_xp, scale=scale,
                    )
                    _stamp_label_np(
                        cell_1, _A1_PATTERN_SMALL, 1, 27, a1_color_np_xp, scale=scale,
                    )

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
    partner_feed_dim=5,
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
        _stamp_label_np, _A0_PATTERN_SMALL, _A1_PATTERN_SMALL,
        _recolour_message_dot,
    )

    img_h = GRID_ROWS * TILE_PIXELS
    img_w = GRID_COLS * TILE_PIXELS
    scale = 20
    padding = 4
    all_video_frames: list = []
    a0_color_np = np.array([255, 140, 0], dtype=np.uint8)
    a1_color_np = np.array([255, 0, 255], dtype=np.uint8)

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
            partner_feed_dim=partner_feed_dim,
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
            _recolour_message_dot(base_0, ep_states[t], 0)
            _recolour_message_dot(base_1, ep_states[t], 1)
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
                box_color = [255, 255, 0] if solved else [255, 255, 255]
                if pick_0_view is not None:
                    _draw_choice_on_cell(cell_0, pick_0_view, 0, scale, color=box_color)
                if pick_1_view is not None:
                    _draw_choice_on_cell(cell_1, pick_1_view, 1, scale, color=box_color)

            _stamp_label_np(cell_0, _A0_PATTERN_SMALL, 1, 27, a0_color_np, scale=scale)
            _stamp_label_np(cell_1, _A1_PATTERN_SMALL, 1, 27, a1_color_np, scale=scale)

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
    partner_feed_dim=5,
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
            partner_feed_dim=partner_feed_dim,
        )




def _log_card_game_per_head_attn_panel(
    inner_env, policy, params, max_steps, tag, video_dir, logger,
    feed_attn_dims=None, ja_card_masks=None,
    partner_feed_dim=5, num_episodes=2,
    rng_seed_base=8000,
    seed_idx=None,
    has_comm=False,
):
    """Per-head x per-step attention overlay PNG, one per agent per episode.

    Mirrors the post-hoc `attention_numbers._save_per_head_action_overlay`:
    NUM_HEADS rows x T columns, each cell = agent-frame obs with that head's
    attention heatmap, plus a red box around the picked card on the decision
    step. Built from the agent's actual recoloured/shuffled obs so the overlay
    lands on the cards the policy actually saw.
    """
    import wandb
    from pathlib import Path
    from evaluation.card_game.attention_numbers import (
        _save_per_head_action_overlay,
    )

    img_h_const, img_w_const = 21, 35

    for ep in range(num_episodes):
        ep_rng = jax.random.PRNGKey(rng_seed_base + ep * 17)
        ep_states, attn_data, ep_actions, ep_messages, ep_obs = run_episode_with_states(
            ep_rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=True,
            collect_obs=True,
            feed_other_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
            partner_feed_dim=partner_feed_dim,
        )

        T = len(ep_obs)
        if T == 0:
            continue
        is_decision_seq = [t == T - 1 for t in range(T)]

        for agent_idx, agent_key in enumerate(("agent_0", "agent_1")):
            maps = attn_data.get(agent_key, [])
            if not maps or len(maps) < T:
                continue

            per_head_seq = []
            for a in maps[:T]:
                a_sq = np.asarray(a).squeeze()
                if a_sq.ndim == 2:
                    a_sq = a_sq[..., None]
                per_head_seq.append(a_sq)
            per_head_seq = np.stack(per_head_seq, axis=0)  # (T, fh, fw, num_heads)

            obs_seq = []
            for o in ep_obs:
                a_obs = np.asarray(o[agent_key])
                img = a_obs[: img_h_const * img_w_const * 3].reshape(
                    img_h_const, img_w_const, 3
                )
                obs_seq.append(np.clip(img, 0.0, 1.0).astype(np.float32))
            obs_seq = np.stack(obs_seq, axis=0)  # (T, img_h, img_w, 3)

            # Pick view-slot at decision step. Translate the recorded raw action
            # through recolouring and position perm exactly as attention_numbers
            # does: inv_recol -> canonical, then argsort(pos_perm) to view-slot.
            action_view_slots = [-1] * T
            if len(ep_actions) >= T and len(ep_states) >= T:
                state_dec = ep_states[T - 1]
                inv_recol = None
                pos_perm = None
                s = state_dec
                while s is not None:
                    if inv_recol is None and hasattr(s, "per_agent_inv_recolouring"):
                        inv_recol = np.asarray(s.per_agent_inv_recolouring[agent_key])
                    if pos_perm is None and hasattr(s, "per_agent_perm"):
                        pos_perm = np.asarray(s.per_agent_perm[agent_key])
                    s = getattr(s, "env_state", None)
                raw = int(ep_actions[T - 1][agent_idx])
                if raw >= 0 and inv_recol is not None and pos_perm is not None:
                    own_pick_gt = int(inv_recol[raw])
                    pos_perm_inv = np.argsort(pos_perm)
                    action_view_slots[T - 1] = int(pos_perm_inv[own_pick_gt])

            partner_msg_view_slots = [-1] * T  # no comm -> no message overlay

            out_path = Path(video_dir) / f"per_head_attn_{agent_key}_ep{ep}.png"
            seed_label = "" if seed_idx is None else f"seed {seed_idx}  "
            agent_label = f"Agent {agent_idx}"
            human_title = (
                f"{seed_label}{agent_label}  episode {ep}  "
                f"(own attention, own obs)"
            )
            legend = (
                "  (red box = own pick at decision step)"
                if not has_comm
                else "  (coloured dot/box = own action; white square = partner msg)"
            )
            _save_per_head_action_overlay(
                per_head_seq=per_head_seq,
                obs_seq=obs_seq,
                action_view_slots=action_view_slots,
                is_decision_seq=is_decision_seq,
                partner_msg_view_slots=partner_msg_view_slots,
                out_path=out_path,
                title=human_title,
                agent_idx=agent_idx,
                legend=legend,
            )
            logger.log(
                {f"{tag}/per_head_attn_{agent_key}_ep{ep}": wandb.Image(str(out_path))},
                commit=False,
            )
            print(f"[card_game] Saved per-head overlay: {out_path}")
