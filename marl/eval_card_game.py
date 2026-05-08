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
    white_border = [255, 255, 255]

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
                _draw_choice_on_cell(cell_0, view_0, 0, scale, color=white_border)
            if view_1 >= 0:
                _draw_choice_on_cell(cell_1, view_1, 1, scale, color=white_border)

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


def _classify_role_pattern(ep_messages, n_delib):
    """Classify a single episode by who first adopts the partner's previous
    message and whether that role is kept.

    Returns one of:
      "a0_follower"  — only A0 ever follows; A1 is the stable proposer.
      "a1_follower"  — only A1 ever follows; A0 is the stable proposer.
      "role_swap"    — both agents follow at some point (negotiation).
      "no_follow"    — no agent ever adopts the partner's previous message
                       (either aligned from start or never aligned at all).
    Also returns (first_follower_step_1indexed_or_None, n_a0_follows, n_a1_follows).
    """
    if n_delib < 2:
        return "no_follow", None, 0, 0
    a0_follows = 0
    a1_follows = 0
    first_follower = None
    first_step = None
    for t in range(1, n_delib):
        prev_0 = int(ep_messages[t - 1][0])
        prev_1 = int(ep_messages[t - 1][1])
        curr_0 = int(ep_messages[t][0])
        curr_1 = int(ep_messages[t][1])
        # When both partners messaged the same colour last step there's
        # nothing to "follow" — skip that step from the role accounting.
        if prev_0 == prev_1 or prev_0 < 0 or prev_1 < 0:
            continue
        a0_followed = (curr_0 == prev_1)
        a1_followed = (curr_1 == prev_0)
        if a0_followed:
            a0_follows += 1
            if first_follower is None:
                first_follower = 0
                first_step = t + 1
        if a1_followed:
            a1_follows += 1
            if first_follower is None:
                first_follower = 1
                first_step = t + 1
    if first_follower is None:
        return "no_follow", None, 0, 0
    if a0_follows > 0 and a1_follows == 0:
        regime = "a0_follower"
    elif a1_follows > 0 and a0_follows == 0:
        regime = "a1_follower"
    else:
        regime = "role_swap"
    return regime, first_step, a0_follows, a1_follows


def _log_card_game_role_dynamics(
    inner_env, policy, params, max_steps, tag, video_dir, logger,
    feed_attn_dims=None, ja_card_masks=None,
    num_episodes=50, rng_seed_base=500,
):
    """Per-episode role-pattern (proposer / follower) distribution across SP eps.

    For each episode, classifies the deliberation dynamics into one of:
      a0_follower / a1_follower / role_swap / no_follow.
    Saves a bar chart and prints aggregate counts.
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = {"a0_follower": 0, "a1_follower": 0, "role_swap": 0, "no_follow": 0}
    first_step_by_regime = {k: [] for k in counts}
    n_delib = max_steps - 1

    success = fail = 0
    for ep in range(num_episodes):
        ep_rng = jax.random.PRNGKey(rng_seed_base + ep)
        _, ep_actions, ep_messages = run_episode_with_states(
            ep_rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=False,
            feed_other_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
        )
        if not ep_actions:
            continue
        pick_0 = int(ep_actions[-1][0])
        pick_1 = int(ep_actions[-1][1])
        if pick_0 < 0 or pick_1 < 0 or pick_0 != pick_1:
            fail += 1
            continue
        success += 1
        regime, first_step, _, _ = _classify_role_pattern(ep_messages, n_delib)
        counts[regime] += 1
        if first_step is not None:
            first_step_by_regime[regime].append(first_step)

    print(f"\n=== {tag} role dynamics ({num_episodes} SP eps) ===")
    print(f"  success/fail: {success}/{fail}")
    for regime in ("a0_follower", "a1_follower", "role_swap", "no_follow"):
        n = counts[regime]
        if n == 0:
            print(f"  {regime:>14s}: {n}")
            continue
        steps = first_step_by_regime[regime]
        if steps:
            steps_arr = np.array(steps)
            print(
                f"  {regime:>14s}: {n}  first-follow step "
                f"mean={steps_arr.mean():.2f} median={int(np.median(steps_arr))}"
            )
        else:
            print(f"  {regime:>14s}: {n}")

    if success == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    regimes = ["a0_follower", "a1_follower", "role_swap", "no_follow"]
    palette = ["#ff8c00", "#ff00ff", "#5b8def", "#888888"]
    values = [counts[r] for r in regimes]
    ax.bar(regimes, values, color=palette, edgecolor="black")
    for i, v in enumerate(values):
        if v > 0:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("episode count")
    ax.set_title(
        f"{tag} — role pattern over {success} successful eps "
        f"(plus {fail} failed)"
    )
    plt.tight_layout()
    os.makedirs(video_dir, exist_ok=True)
    png_path = os.path.join(video_dir, "role_dynamics.png")
    plt.savefig(png_path)
    plt.close(fig)
    try:
        import wandb
        logger.log(
            {f"{tag}/role_dynamics": wandb.Image(png_path)},
            commit=False,
        )
    except Exception:
        pass


def _log_card_game_coordination_dynamics(
    inner_env, policy, params, max_steps, tag, video_dir, logger,
    feed_attn_dims=None, ja_card_masks=None,
    num_episodes=50, rng_seed_base=400,
):
    """Per-episode "lock-in step" histogram across `num_episodes` SP eps.

    For each successful episode (= matching decision picks), find the latest
    step from which both agents' messages already equalled the eventual pick
    *and* stayed equal until decision. Earlier values mean the protocol
    converged fast (proposer-follower); values close to max_steps mean late
    convergence (negotiation / oscillation). Failed episodes are tallied
    separately.
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lock_in_steps = []
    failed = 0
    n_delib = max_steps - 1

    for ep in range(num_episodes):
        ep_rng = jax.random.PRNGKey(rng_seed_base + ep)
        _, ep_actions, ep_messages = run_episode_with_states(
            ep_rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=False,
            feed_other_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
        )
        if not ep_actions:
            continue
        pick_0 = int(ep_actions[-1][0])
        pick_1 = int(ep_actions[-1][1])
        if pick_0 < 0 or pick_1 < 0 or pick_0 != pick_1:
            failed += 1
            continue
        target = pick_0
        deliberation = ep_messages[:n_delib]
        # Walk backwards from the last deliberation step, count consecutive
        # steps where both messages == target.
        lock_in = n_delib + 1   # 1-indexed: "decision step itself"
        for t in range(n_delib - 1, -1, -1):
            if (int(deliberation[t][0]) == target and
                int(deliberation[t][1]) == target):
                lock_in = t + 1
            else:
                break
        lock_in_steps.append(lock_in)

    n_success = len(lock_in_steps)
    print(f"\n=== {tag} coordination dynamics ({num_episodes} SP eps) ===")
    print(f"  successful: {n_success}/{num_episodes}")
    print(f"  failed:     {failed}/{num_episodes}")
    if not lock_in_steps:
        return
    arr = np.array(lock_in_steps)
    print(
        f"  lock-in step (1-indexed): "
        f"mean={arr.mean():.2f}  median={int(np.median(arr))}  "
        f"min={int(arr.min())}  max={int(arr.max())}"
    )
    # Per-step counts.
    counts = np.bincount(arr, minlength=max_steps + 2)[1:max_steps + 2]
    print("  per-step lock-in distribution:")
    for s, c in enumerate(counts, start=1):
        if c == 0:
            continue
        marker = " (decision)" if s == max_steps else ""
        print(f"    step {s}: {c:>4d}  ({100*c/n_success:>5.1f}%){marker}")

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0.5, max_steps + 1.5)
    ax.hist(arr, bins=bins, edgecolor="black", color="#5b8def")
    ax.set_xlabel("step at which messages first locked in to the decision")
    ax.set_ylabel("episode count")
    ax.set_title(
        f"{tag} — coordination lock-in step ({n_success}/{num_episodes} eps)"
    )
    ax.set_xticks(np.arange(1, max_steps + 1))
    plt.tight_layout()
    os.makedirs(video_dir, exist_ok=True)
    png_path = os.path.join(video_dir, "coordination_lock_in.png")
    plt.savefig(png_path)
    plt.close(fig)
    try:
        import wandb
        logger.log(
            {f"{tag}/coordination_lock_in": wandb.Image(png_path)},
            commit=False,
        )
    except Exception:
        pass


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
    print()


def _log_card_game_eval_video(inner_env, policy, params, max_steps, tag, video_dir, logger,
                               feed_attn_dims=None,
                               ja_card_masks=None,
                               num_episodes=30, fps=3):
    """Run multiple card game episodes and save a video with attention spots and choices."""
    import wandb
    from envs.card_game.rendering import (
        render_card_game_minimal,
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
                white_border = [255, 255, 255]
                if view_0 >= 0:
                    _draw_choice_on_cell(cell_0, view_0, 0, scale, color=white_border)
                if view_1 >= 0:
                    _draw_choice_on_cell(cell_1, view_1, 1, scale, color=white_border)

            _stamp_label_np(cell_0, _A0_PATTERN_SMALL, 1, 27, a0_color_np, scale=scale)
            _stamp_label_np(cell_1, _A1_PATTERN_SMALL, 1, 27, a1_color_np, scale=scale)

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
                ep_states, attn_data, ep_actions, ep_messages, ep_obs = run_episode_with_states(
                    ep_rng, inner_env, params_i, policy,
                    params_j, policy, max_steps,
                    collect_attention=True,
                    collect_obs=True,
                    feed_other_attn_dims=feed_attn_dims,
                    ja_card_masks=ja_card_masks,
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
                        white_border = [255, 255, 255]
                        if view_0 >= 0:
                            _draw_choice_on_cell(cell_0, view_0, 0, scale, color=white_border)
                        if view_1 >= 0:
                            _draw_choice_on_cell(cell_1, view_1, 1, scale, color=white_border)

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
