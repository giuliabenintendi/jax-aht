"""Post-hoc per-agent attention analysis for the card game.

Loads a saved training checkpoint, runs N eval episodes with a chosen seed,
and for each agent produces two separate figures per episode:
  - <agent>_obs.png: agent's own-frame view across the episode, with a dot
    highlighting the agent's own message (or pick, on the decision step) in
    its own colour (orange for agent 0, magenta for agent 1).
  - <agent>_attention.png: attention map per step rendered with the agent's
    palette (Oranges / RdPu), normalised to the episode's global max.

Usage:
    ./run_gpu.sh <gpu> evaluation.analyze_attention \\
        --checkpoint /path/to/saved/train_run \\
        --seed-idx 0 \\
        --num-episodes 5 \\
        --output-dir plots/card_game/<run_name>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import (
    AGENT_0_COLOR,
    AGENT_1_COLOR,
    CARD_COLORS,
    GRID_COLS,
    GRID_ROWS,
    NUM_CARDS,
    TILE_PIXELS,
    render_card_game,
)
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import run_episode_with_states


_H = GRID_ROWS * TILE_PIXELS
_W = GRID_COLS * TILE_PIXELS
_WHITE = np.array([255, 255, 255], dtype=np.uint8)
_AGENT_GRID_POSITIONS = [(0, 2), (2, 2)]  # (row, col) of agent 0, agent 1 tiles


def _draw_border_np(img, row, col, tile_size, color):
    """Draw 1-pixel border around the tile at (row, col)."""
    y = row * tile_size
    x = col * tile_size
    img[y, x:x + tile_size] = color
    img[y + tile_size - 1, x:x + tile_size] = color
    img[y:y + tile_size, x] = color
    img[y:y + tile_size, x + tile_size - 1] = color
    return img


def _walk_to_card_state(state):
    """Walk .env_state chain to the innermost CardGameState."""
    s = state
    while hasattr(s, "env_state") and not hasattr(s, "card_permutation"):
        s = s.env_state
    return s


def _get_per_agent_info(state, agent_idx: int):
    """Return (card_permutation, pos_perm, recolouring) for agent_idx from a wrapped state."""
    name = f"agent_{agent_idx}"
    recol = np.asarray(state.per_agent_recolouring[name])
    pos_perm = np.asarray(state.env_state.per_agent_perm[name])
    card_state = _walk_to_card_state(state)
    card_perm = np.asarray(card_state.card_permutation)
    return card_perm, pos_perm, recol


def _render_own_frame(card_perm, pos_perm, recolouring, agent_idx: int,
                     own_gt_value: int, is_decision: bool):
    """Render agent's own-frame view (post-OP) with the agent's own-colour
    marker: a dot on its messaged card (deliberation), or an external bounding
    box around its picked card (decision step)."""
    # render_card_game uses jax.lax.scan, so input must be a JAX array
    base = np.asarray(render_card_game(jnp.asarray(card_perm))).copy()
    TP = TILE_PIXELS

    # Position shuffle of the card row
    card_row = base[TP:2 * TP, :, :].copy()
    tiles = card_row.reshape(TP, NUM_CARDS, TP, 3)
    shuffled = tiles[:, pos_perm, :, :]
    base[TP:2 * TP, :, :] = shuffled.reshape(TP, NUM_CARDS * TP, 3)

    # Recolouring of the card row: read from the original source, write to a
    # separate destination so permutation cycles don't cascade through each other.
    src = base[TP:2 * TP, :, :]
    dst = src.copy()
    card_colors_np = np.asarray(CARD_COLORS)
    for gt_idx in range(NUM_CARDS):
        original = card_colors_np[gt_idx]
        new_color = card_colors_np[int(recolouring[gt_idx])]
        mask = np.all(src == original, axis=-1)
        dst[mask] = new_color
    base[TP:2 * TP, :, :] = dst

    # Ego highlight — white border around this agent's tile
    agent_row, agent_col = _AGENT_GRID_POSITIONS[agent_idx]
    base = _draw_border_np(base, agent_row, agent_col, TP, _WHITE)

    # Decision indicator (white 4×4 top-left)
    if is_decision:
        base[:4, :4, :] = _WHITE

    # Agent's own-colour marker on the card it acted on this step
    if own_gt_value >= 0:
        gt_positions = np.where(card_perm == own_gt_value)[0]
        if len(gt_positions):
            gt_col = int(gt_positions[0])
            view_positions = np.where(pos_perm == gt_col)[0]
            if len(view_positions):
                view_col = int(view_positions[0])
                agent_color = np.asarray(
                    AGENT_0_COLOR if agent_idx == 0 else AGENT_1_COLOR
                )
                if is_decision:
                    base = _draw_border_np(base, 1, view_col, TP, agent_color)
                else:
                    dot_size = 2
                    dy = TP + (TP - dot_size) // 2
                    dx = view_col * TP + (TP - dot_size) // 2
                    base[dy:dy + dot_size, dx:dx + dot_size] = agent_color

    return base


def _render_joint_canonical_sequence(
    ep_states, ep_actions, ep_messages, output_path: Path, max_steps: int,
):
    """1 × T row of canonical-frame snapshots showing BOTH agents' actions.

    Cards are rendered in canonical colours and canonical positions (no
    per-agent recolouring or position shuffle). At each step:
      - Agent 0's canonical action is marked in orange.
      - Agent 1's canonical action is marked in magenta.
      - Marks are placed in different vertical halves of the card so they
        don't overlap when both agents act on the same canonical card.
    Decision step: the picked card gets a coloured border per agent
    (agent 0 outer, agent 1 inset by 1 pixel).
    """
    num_steps = min(len(ep_states) - 1, max_steps)
    if num_steps == 0:
        return

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.8, 2.0))
    if num_steps == 1:
        axes = [axes]

    TP = TILE_PIXELS
    card_row = 1
    a0_color = np.asarray(AGENT_0_COLOR)
    a1_color = np.asarray(AGENT_1_COLOR)

    def _canonical_pos_of(card_perm: np.ndarray, canonical_value: int) -> int:
        if canonical_value < 0:
            return -1
        positions = np.where(card_perm == canonical_value)[0]
        return int(positions[0]) if positions.size > 0 else -1

    def _draw_dot_top(img, pos: int, color):
        dot_size = 2
        dy = card_row * TP + 1
        dx = pos * TP + (TP - dot_size) // 2
        img[dy:dy + dot_size, dx:dx + dot_size] = color
        return img

    def _draw_dot_bottom(img, pos: int, color):
        dot_size = 2
        dy = card_row * TP + TP - 1 - dot_size
        dx = pos * TP + (TP - dot_size) // 2
        img[dy:dy + dot_size, dx:dx + dot_size] = color
        return img

    def _draw_border_inset(img, pos: int, color, inset: int):
        y = card_row * TP + inset
        x = pos * TP + inset
        size = TP - 2 * inset
        img[y, x:x + size] = color
        img[y + size - 1, x:x + size] = color
        img[y:y + size, x] = color
        img[y:y + size, x + size - 1] = color
        return img

    for t in range(num_steps):
        is_decision = (t == num_steps - 1)
        state = ep_states[t]
        card_state = _walk_to_card_state(state)
        card_perm = np.asarray(card_state.card_permutation)

        a0_inv = np.asarray(state.per_agent_inv_recolouring["agent_0"])
        a1_inv = np.asarray(state.per_agent_inv_recolouring["agent_1"])

        if is_decision:
            a0_own = int(ep_actions[t][0]) if t < len(ep_actions) else -1
            a1_own = int(ep_actions[t][1]) if t < len(ep_actions) else -1
        else:
            a0_own = int(ep_messages[t][0]) if t < len(ep_messages) else -1
            a1_own = int(ep_messages[t][1]) if t < len(ep_messages) else -1
        a0_canon = int(a0_inv[a0_own]) if a0_own >= 0 else -1
        a1_canon = int(a1_inv[a1_own]) if a1_own >= 0 else -1

        img = np.asarray(render_card_game(jnp.asarray(card_perm))).copy()

        a0_pos = _canonical_pos_of(card_perm, a0_canon)
        a1_pos = _canonical_pos_of(card_perm, a1_canon)

        if is_decision:
            if a0_pos >= 0:
                img = _draw_border_inset(img, a0_pos, a0_color, inset=0)
            if a1_pos >= 0:
                img = _draw_border_inset(img, a1_pos, a1_color, inset=1)
        else:
            if a0_pos >= 0:
                img = _draw_dot_top(img, a0_pos, a0_color)
            if a1_pos >= 0:
                img = _draw_dot_bottom(img, a1_pos, a1_color)

        axes[t].imshow(img, interpolation="nearest")
        axes[t].axis("off")
        title = "decision" if is_decision else f"step {t}"
        axes[t].set_title(title, fontsize=9)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.05)
    fig.suptitle(
        "Joint canonical view (orange = agent 0, magenta = agent 1; "
        "top dot = agent 0, bottom dot = agent 1; "
        "matching positions = joint behaviour)",
        fontsize=10,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _render_obs_sequence(ep_states, ep_actions, ep_messages, agent_idx: int,
                        output_path: Path, max_steps: int):
    """1 row × T columns: agent's own view with own-action dot per step."""
    num_steps = min(len(ep_states) - 1, max_steps)
    if num_steps == 0:
        return

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.8, 2.0))
    if num_steps == 1:
        axes = [axes]

    for t in range(num_steps):
        is_decision = (t == num_steps - 1)
        if is_decision:
            own_value = int(ep_actions[t][agent_idx]) if t < len(ep_actions) else -1
        else:
            own_value = int(ep_messages[t][agent_idx]) if t < len(ep_messages) else -1

        state = ep_states[t]
        card_perm, pos_perm, recolouring = _get_per_agent_info(state, agent_idx)

        img = _render_own_frame(
            card_perm, pos_perm, recolouring, agent_idx, own_value, is_decision,
        )
        axes[t].imshow(img, interpolation="nearest")
        axes[t].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.05)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _render_attention_sequence(attn_maps, ep_states, ep_actions, ep_messages,
                              agent_idx: int, agent_key: str, cmap: str,
                              output_path: Path, max_steps: int,
                              draw_heatmap: bool = True):
    """1 row × T columns: per-step attention + arrow to messaged card or pick bbox.

    When draw_heatmap is False, the heatmap is omitted (and so is the
    colorbar). Only the card outlines, partner-message dots, own-arrow / pick
    rectangle remain — useful for reading the message dynamics on a clean
    background.
    """
    num_steps = min(len(attn_maps[agent_key]), max_steps)
    if num_steps == 0:
        return

    def _head_average(raw):
        a = np.asarray(raw).squeeze()
        if a.ndim == 3:
            a = a.mean(axis=-1)
        return a

    attn_stack = np.array([_head_average(attn_maps[agent_key][t]) for t in range(num_steps)])
    attn_max = float(attn_stack.max())
    if attn_max <= 0:
        attn_max = 1.0

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.8, 2.0))
    if num_steps == 1:
        axes = [axes]

    extent = (0, _W, _H, 0)
    TP = TILE_PIXELS

    CARD_INSET = 1
    CARD_SIZE = TP - 2 * CARD_INSET  # 5

    agent_rgb = (
        np.asarray(AGENT_0_COLOR if agent_idx == 0 else AGENT_1_COLOR) / 255.0
    )
    partner_rgb = (
        np.asarray(AGENT_1_COLOR if agent_idx == 0 else AGENT_0_COLOR) / 255.0
    )
    # Arrow is always straight vertical: starts from the vertical centre of the
    # agent's row on the X-column of the messaged card, points toward the card.
    agent_row, _agent_col = _AGENT_GRID_POSITIONS[agent_idx]
    arrow_start_y = agent_row * TP + TP / 2.0

    def _add_card_outlines(ax):
        for card_col in range(NUM_CARDS):
            x = card_col * TP + CARD_INSET
            y = 1 * TP + CARD_INSET
            ax.add_patch(patches.Rectangle(
                (x, y), CARD_SIZE, CARD_SIZE,
                linewidth=0.8, edgecolor="black", facecolor="none",
            ))

    def _view_col_of(card_perm, pos_perm, gt_value):
        if gt_value < 0:
            return None
        gt_cols = np.where(card_perm == gt_value)[0]
        if len(gt_cols) == 0:
            return None
        gt_col = int(gt_cols[0])
        view_cols = np.where(pos_perm == gt_col)[0]
        if len(view_cols) == 0:
            return None
        return int(view_cols[0])

    for t in range(num_steps):
        is_decision = (t == num_steps - 1)
        if draw_heatmap:
            attn = _head_average(attn_maps[agent_key][t])
            axes[t].imshow(
                attn, cmap=cmap, vmin=0.0, vmax=attn_max,
                interpolation="nearest", extent=extent,
            )
        _add_card_outlines(axes[t])

        # Overlay action indicator
        state = ep_states[t]
        card_perm, pos_perm, _ = _get_per_agent_info(state, agent_idx)

        # Partner's message in two timing states:
        #   - solid: DELIVERED to this agent at step t (in obs[t], hence what
        #     the attention map actually attended to). That is sent-at-(t-1),
        #     which lives in ep_states[t].messages.
        #   - faded: JUST SENT at step t (not yet in this agent's obs; will be
        #     delivered next step). Sourced from ep_messages[t]. Drawn faded
        #     and only when it differs from the delivered position.
        card_state = _walk_to_card_state(state)
        partner_msg_delivered_gt = int(card_state.messages[1 - agent_idx])
        partner_view_col_delivered = _view_col_of(
            card_perm, pos_perm, partner_msg_delivered_gt,
        )
        partner_view_col_concurrent = None
        if not is_decision and t < len(ep_messages):
            partner_msg_concurrent_gt = int(ep_messages[t][1 - agent_idx])
            partner_view_col_concurrent = _view_col_of(
                card_perm, pos_perm, partner_msg_concurrent_gt,
            )
        # Draw faded first so the solid dot occludes it when they overlap.
        if (
            partner_view_col_concurrent is not None
            and partner_view_col_concurrent != partner_view_col_delivered
        ):
            cx = partner_view_col_concurrent * TP + TP / 2.0
            cy = 1 * TP + TP / 2.0
            axes[t].add_patch(patches.Circle(
                (cx, cy), radius=1.2,
                facecolor=partner_rgb, edgecolor="none", alpha=0.35,
            ))
        if partner_view_col_delivered is not None:
            cx = partner_view_col_delivered * TP + TP / 2.0
            cy = 1 * TP + TP / 2.0
            axes[t].add_patch(patches.Circle(
                (cx, cy), radius=1.2,
                facecolor=partner_rgb, edgecolor="none", alpha=1.0,
            ))

        if is_decision:
            pick_gt = int(ep_actions[t][agent_idx]) if t < len(ep_actions) else -1
            view_col = _view_col_of(card_perm, pos_perm, pick_gt)
            if view_col is not None:
                # External to the 5x5 card — use full tile footprint (7x7)
                axes[t].add_patch(patches.Rectangle(
                    (view_col * TP, 1 * TP),
                    TP, TP,
                    linewidth=1.8, edgecolor=agent_rgb, facecolor="none",
                ))
        else:
            msg_gt = int(ep_messages[t][agent_idx]) if t < len(ep_messages) else -1
            view_col = _view_col_of(card_perm, pos_perm, msg_gt)
            if view_col is not None:
                # Straight vertical arrow on the card's X column; tip lands on
                # the card centre.
                card_center = (
                    view_col * TP + TP / 2.0,
                    1 * TP + TP / 2.0,
                )
                arrow_start = (card_center[0], arrow_start_y)
                axes[t].annotate(
                    "",
                    xy=card_center,
                    xytext=arrow_start,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="red",
                        lw=1.8,
                        shrinkA=0.0,
                        shrinkB=0.0,
                        mutation_scale=14,
                    ),
                )

        axes[t].set_xlim(0, _W)
        axes[t].set_ylim(_H, 0)
        axes[t].axis("off")

    fig.subplots_adjust(left=0.01, right=0.93, top=0.98, bottom=0.02, wspace=0.05)

    if draw_heatmap:
        # Colorbar aligned to the actual rendered axis bbox (imshow shrinks the
        # subplot to preserve aspect, so we need the post-layout position).
        fig.canvas.draw()
        axis_bbox = axes[-1].get_position()
        cbar_ax = fig.add_axes([
            axis_bbox.x1 + 0.01, axis_bbox.y0, 0.008, axis_bbox.height,
        ])
        cbar = fig.colorbar(axes[0].images[0], cax=cbar_ax)
        cbar.set_ticks([0.0, attn_max])
        cbar.ax.tick_params(labelsize=6, length=0)
        cbar.outline.set_linewidth(0.3)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _get_obs_type(alg_config):
    return alg_config.get(
        "OBS_TYPE",
        alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved train run checkpoint directory")
    parser.add_argument("--seed-idx", type=int, default=0,
                        help="Which seed to analyze (0..NUM_SEEDS-1). Ignored when --all-seeds is set.")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Analyze every seed in the checkpoint; output goes to {output_dir}/seed_{i}/ per seed.")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--output-dir", default="plots/card_game")
    parser.add_argument("--episode-rng-base", type=int, default=100,
                        help="Base seed for per-episode RNGs: key = base + ep")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip the attention heatmap and colorbar in the *_attention.png; "
                             "keep card outlines, partner-message dots, and own arrow / pick.")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
    config_path = None
    cur = run_dir
    for _ in range(4):
        candidate = cur / ".hydra" / "config.yaml"
        if candidate.exists():
            config_path = candidate
            break
        cur = cur.parent
    if config_path is None:
        raise FileNotFoundError(
            f"Could not locate .hydra/config.yaml near {run_dir}"
        )
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg_config = cfg["algorithm"]

    if alg_config.get("COMMUNICATION", False):
        env_kwargs = dict(alg_config["ENV_KWARGS"])
        env_kwargs["communication"] = True
        alg_config["ENV_KWARGS"] = env_kwargs

    env = make_env(alg_config["ENV_NAME"], alg_config["ENV_KWARGS"])
    env = LogWrapper(env)
    inner_env = env._env
    max_steps = alg_config["ENV_KWARGS"].get("max_steps", 8)

    obs_type = _get_obs_type(alg_config)
    init_fn = (
        initialize_ja_image_agent if obs_type in ("image", "fov")
        else initialize_ja_agent
    )
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(alg_config, env, rng)

    # JA_CARD_ATTN policies expect a 5-dim translated-partner-attention vector
    # appended to the obs; FEED_OTHER_ATTN policies expect a 4th obs channel.
    # Both must be passed through to run_episode_with_states or the policy
    # gets a malformed input and init/apply shape-checks fail.
    feed_attn = alg_config.get("FEED_OTHER_ATTN", False)
    ja_card_attn = alg_config.get("JA_CARD_ATTN", False)
    # The 5-dim partner-attention concat only happens when JA_CARD_PARTNER_FEED
    # is on (default true for BC). Match training, otherwise the network's
    # input shape is mis-sized.
    ja_card_partner_feed = ja_card_attn and alg_config.get("JA_CARD_PARTNER_FEED", True)
    feed_attn_dims = None
    ja_card_masks = None
    if feed_attn or ja_card_partner_feed:
        from agents.ja_image_actor_critic import _compute_resnet_output_dims
        img_h = inner_env.grid_height * inner_env.tile_size
        img_w = inner_env.grid_width * inner_env.tile_size
        feat_h, feat_w = _compute_resnet_output_dims(
            img_h, img_w,
            stride=alg_config.get("CONV_STRIDE", 2),
            kernel_size=alg_config.get("CONV_KERNEL_SIZE", 3),
            padding=alg_config.get("CONV_PADDING", "SAME"),
            num_blocks=alg_config.get("CONV_NUM_BLOCKS", 4),
        )
        if feed_attn:
            feed_attn_dims = (img_h, img_w, feat_h, feat_w)
        if ja_card_partner_feed:
            from agents.ja_utils import build_card_masks
            ja_card_masks = build_card_masks(img_h, img_w, feat_h, feat_w)

    run_data = load_train_run(str(ckpt_path))
    final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]

    # Per-seed best-checkpoint training-chunk return is precomputed by
    # _select_best_per_seed_ckpt and saved under "per_ckpt_chunk_return" /
    # "best_ckpt_idx". Use it to label seed dirs so the on-disk layout makes
    # per-seed performance scannable at a glance, with no extra rollouts.
    per_ckpt_returns = run_data.get("per_ckpt_chunk_return")
    best_ckpt_idx = run_data.get("best_ckpt_idx")
    best_seed_returns = None
    if per_ckpt_returns is not None and best_ckpt_idx is not None:
        per_ckpt_returns_np = np.asarray(per_ckpt_returns)
        best_idx_np = np.asarray(best_ckpt_idx)
        best_seed_returns = per_ckpt_returns_np[
            np.arange(per_ckpt_returns_np.shape[0]), best_idx_np
        ]

    if args.all_seeds:
        seed_indices = list(range(num_seeds))
    else:
        if args.seed_idx >= num_seeds:
            raise ValueError(
                f"seed_idx={args.seed_idx} out of range for {num_seeds} seeds"
            )
        seed_indices = [args.seed_idx]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  seeds in checkpoint: {num_seeds}, analyzing seeds: {seed_indices}")
    print(f"  max_steps: {max_steps}")
    print(f"  saving to: {output_dir.resolve()}")

    for seed_idx in seed_indices:
        params = jax.tree.map(lambda x, _i=seed_idx: x[_i], final_params)
        if args.all_seeds:
            name = f"seed_{seed_idx}"
            if best_seed_returns is not None:
                name += f"_r{float(best_seed_returns[seed_idx]):.3f}"
            seed_out = output_dir / name
        else:
            seed_out = output_dir
        seed_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[seed {seed_idx}] -> {seed_out.resolve()}")

        for ep in range(args.num_episodes):
            ep_rng = jax.random.PRNGKey(args.episode_rng_base + ep)
            ep_states, attn_maps, ep_actions, ep_messages = run_episode_with_states(
                ep_rng, inner_env, params, policy, params, policy, max_steps,
                collect_attention=True,
                feed_other_attn_dims=feed_attn_dims,
                ja_card_masks=ja_card_masks,
            )

            ep_dir = seed_out / f"episode_{ep}"
            ep_dir.mkdir(exist_ok=True)

            for agent_idx, (agent_key, cmap) in enumerate([
                ("agent_0", "Oranges"),
                ("agent_1", "RdPu"),
            ]):
                _render_obs_sequence(
                    ep_states, ep_actions, ep_messages, agent_idx,
                    output_path=ep_dir / f"{agent_key}_obs.png",
                    max_steps=max_steps,
                )
                _render_attention_sequence(
                    attn_maps, ep_states, ep_actions, ep_messages,
                    agent_idx, agent_key, cmap,
                    output_path=ep_dir / f"{agent_key}_attention.png",
                    max_steps=max_steps,
                    draw_heatmap=not args.no_heatmap,
                )
            _render_joint_canonical_sequence(
                ep_states, ep_actions, ep_messages,
                output_path=ep_dir / "joint_canonical.png",
                max_steps=max_steps,
            )

            summary = ep_dir / "summary.txt"
            with summary.open("w") as f:
                f.write(f"Episode {ep}\n")
                f.write(f"  seed_idx: {seed_idx}\n")
                f.write(f"  max_steps: {max_steps}\n")
                f.write(f"  ep_actions (GT): {ep_actions}\n")
                f.write(f"  ep_messages (GT): {ep_messages}\n")

            print(f"  seed {seed_idx} episode {ep}: saved to {ep_dir}/")

    print(f"\nDone. Figures in {output_dir.resolve()}/")


if __name__ == "__main__":
    main()
