"""Visualize head-averaged attention over time for a trained JA checkpoint.

For each of N self-play episodes, run the policy and save one compact strip
per agent. Each strip shows the agent-view observation at every timestep with
the head-averaged attention map overlaid.

The file also keeps `_save_per_head_action_overlay`, which is reused by
`marl.eval_card_game` for training-time visualization.

Usage:
    ./run_gpu.sh 5 evaluation.card_game.attention_numbers \\
        --checkpoint /scratch/.../saved_train_run \\
        --seed-idx 0 \\
        --num-episodes 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.card_game.rendering import (
    AGENT_0_COLOR, AGENT_1_COLOR,
    _draw_card_border_upscaled,
    CARD_COLORS, NUM_CARDS, TILE_PIXELS, render_card_game_minimal,
)
from agents.ja_utils import build_card_masks
from evaluation.card_game._card_game_utils import load_card_game_eval
from evaluation.vis_episodes import run_episode_with_states


def _walk_to_card_state(state):
    s = state
    while hasattr(s, "env_state") and not hasattr(s, "card_permutation"):
        s = s.env_state
    return s


def _render_agent_view(state, agent_key: str, step_count: int) -> np.ndarray:
    """Reconstruct the (21, 35, 3) obs as agent_key actually saw it.

    Uses render_card_game_minimal as the base (cards + timestep counter, no
    agent indicators), then applies OP position-shuffle and recolouring.
    Skips the partner-message dot (it would require digging out the previous
    step's messages; this script is focused on the raw attention panels).
    """
    card_state = _walk_to_card_state(state)
    card_perm = np.asarray(card_state.card_permutation)
    pos_perm = np.asarray(state.env_state.per_agent_perm[agent_key])
    recolouring = np.asarray(state.per_agent_recolouring[agent_key])

    base = np.asarray(render_card_game_minimal(
        jnp.asarray(card_perm), jnp.int32(step_count)
    )).copy()

    TP = TILE_PIXELS
    card_row = base[TP:2 * TP, :, :].copy()
    tiles = card_row.reshape(TP, NUM_CARDS, TP, 3)
    shuffled = tiles[:, pos_perm, :, :]
    base[TP:2 * TP, :, :] = shuffled.reshape(TP, NUM_CARDS * TP, 3)

    src = base[TP:2 * TP, :, :]
    dst = src.copy()
    card_colors_np = np.asarray(CARD_COLORS)
    for gt_idx in range(NUM_CARDS):
        original = card_colors_np[gt_idx]
        new_color = card_colors_np[int(recolouring[gt_idx])]
        mask = np.all(src == original, axis=-1)
        dst[mask] = new_color
    base[TP:2 * TP, :, :] = dst

    return base.astype(np.float32) / 255.0


def _draw_own_action_dot_raw(base: np.ndarray, view_col: int, agent_idx: int) -> np.ndarray:
    """Draw a 2x2 dot of the agent's own colour at the chosen card."""
    if view_col < 0:
        return base
    color = np.asarray(AGENT_0_COLOR if agent_idx == 0 else AGENT_1_COLOR, dtype=np.uint8)
    cx = int(view_col * TILE_PIXELS + TILE_PIXELS / 2)
    cy = int(TILE_PIXELS + TILE_PIXELS / 2)
    base[max(0, cy - 1): cy + 1, max(0, cx - 1): cx + 1] = color
    return base


def _save_per_head_action_overlay(
    per_head_seq: np.ndarray,
    obs_seq: np.ndarray,
    action_view_slots: list,
    is_decision_seq: list,
    partner_msg_view_slots: list,
    out_path: Path,
    title: str,
    agent_idx: int,
    img_h: int = 21,
    img_w: int = 35,
    legend: str | None = None,
    row_labels: list[str] | None = None,
    row_cmaps: list[str] | None = None,
    row_card_values: list | None = None,
    row_action_slots: list | None = None,
    row_marker_colors: list | None = None,
) -> None:
    """4 rows (one per head) × T columns; obs as background, head-specific
    attention overlay, and a colored marker on the card the agent acted on
    (dot for deliberation message, box for decision pick).

    Pass `row_labels` (length H) to override the default "head 0", "head 1"… —
    e.g. ["avg"] for a single head-averaged row.

    Pass `row_card_values` (length H, each None or a (T, 5) array) to render a
    row as per-card numbers instead of a heatmap — the value is printed on each
    of the 5 card cells. Used for the partner-feed row, which is just 5 numbers
    per step and is clearer as text than as a blocky synthetic heatmap.

    Args:
        per_head_seq: (T, fh, fw, num_heads).
        obs_seq: (T, img_h, img_w, 3) agent-frame obs, float [0, 1].
        action_view_slots: per step int in [0, NUM_CARDS) or -1 if invalid.
        is_decision_seq: per step bool — True at the decision step.
        out_path: PNG.
        agent_idx: 0 (orange marker) or 1 (magenta).
    """
    T, fh, fw, H = per_head_seq.shape
    fig, axes = plt.subplots(H, T, figsize=(2.0 * T, 1.6 * H + 0.4))
    if H == 1:
        axes = axes[None, :]
    if T == 1:
        axes = axes[:, None]

    marker_color = "#ff8c00" if agent_idx == 0 else "#ff00ff"
    TP = 7  # TILE_PIXELS
    card_y_lo, card_y_hi = 7, 14

    NUM_CARDS = 5
    for h_idx in range(H):
        card_values = (
            row_card_values[h_idx]
            if (row_card_values and h_idx < len(row_card_values))
            else None
        )
        # Per-row action markers: each row draws only its own slots in its own
        # colour (e.g. own pick on the own row, partner pick on the partner
        # row). Falls back to the shared action_view_slots / marker_color.
        slots_h = (
            row_action_slots[h_idx]
            if (row_action_slots and h_idx < len(row_action_slots))
            else action_view_slots
        )
        color_h = (
            row_marker_colors[h_idx]
            if (row_marker_colors and h_idx < len(row_marker_colors))
            else marker_color
        )
        for t in range(T):
            ax = axes[h_idx, t]
            ax.imshow(obs_seq[t])
            if card_values is not None:
                # Render this row as per-card numbers instead of a heatmap.
                vals = np.asarray(card_values[t])
                vmax = float(vals.max())
                # vmax < 1e-9 → no feed yet (t=0: partner_feed is all zeros);
                # the empty range skips drawing any numbers for that step.
                for c in (range(NUM_CARDS) if vmax >= 1e-9 else ()):
                    cx = c * TP + TP / 2
                    cy = (card_y_lo + card_y_hi) / 2
                    v = float(vals[c])
                    # bold the dominant card so the argmax is easy to spot
                    is_peak = (vmax > 1e-6) and (v >= vmax - 1e-6)
                    ax.text(
                        cx, cy, f"{v:.2f}",
                        ha="center", va="center", fontsize=5.5,
                        color="black",
                        fontweight="bold" if is_peak else "normal",
                        bbox=dict(
                            boxstyle="round,pad=0.1",
                            facecolor="yellow" if is_peak else "white",
                            edgecolor="none", alpha=0.75,
                        ),
                    )
            else:
                attn = per_head_seq[t, :, :, h_idx]
                attn_up = jax.image.resize(jnp.asarray(attn), (img_h, img_w), method="bilinear")
                # Per-cell normalization: with global vmax, step 0 (near-uniform
                # attention from a fresh LSTM state, ~0.02/cell) is invisible
                # against a peaked-1.0 cell elsewhere in the grid. Normalizing per
                # cell makes the spatial pattern visible at every step regardless
                # of magnitude. Trade-off: intensities are NOT comparable across
                # cells; treat each cell as a relative heatmap.
                cell_vmax = float(np.asarray(attn_up).max())
                if cell_vmax < 1e-6:
                    cell_vmax = 1.0
                cmap_h = row_cmaps[h_idx] if (row_cmaps and h_idx < len(row_cmaps)) else "hot"
                # Alpha proportional to attention value (peak alpha 0.85). Low-attention
                # regions become transparent so colormaps with white-at-low (e.g. RdPu)
                # don't wash out the obs background.
                attn_up_np = np.asarray(attn_up)
                attn_norm = np.clip(attn_up_np / cell_vmax, 0.0, 1.0)
                rgba = plt.get_cmap(cmap_h)(attn_norm)
                rgba[..., 3] = attn_norm * 0.85
                ax.imshow(rgba,
                          extent=(-0.5, img_w - 0.5, img_h - 0.5, -0.5))

            slot = slots_h[t]
            if slot is not None and slot >= 0:
                x0 = slot * TP - 0.5
                y0 = card_y_lo - 0.5
                if is_decision_seq[t]:
                    rect = plt.Rectangle(
                        (x0, y0), TP, card_y_hi - card_y_lo,
                        fill=False, edgecolor=color_h, linewidth=2.0,
                    )
                    ax.add_patch(rect)
                else:
                    cx = x0 + TP / 2
                    cy = y0 + (card_y_hi - card_y_lo) / 2
                    circ = plt.Circle(
                        (cx, cy), radius=1.5,
                        facecolor=color_h, edgecolor="black", linewidth=0.4,
                    )
                    ax.add_patch(circ)

            # Partner's message dot: white square at the messaged card's view slot
            # (the env actually renders this in the obs as a 2x2 white dot, but our
            # simplified renderer skips it; we draw it as an overlay here so it shows
            # through the heat layer).
            pslot = partner_msg_view_slots[t]
            if pslot is not None and pslot >= 0:
                p_x0 = pslot * TP - 0.5
                p_y0 = card_y_lo - 0.5
                pcx = p_x0 + TP / 2
                pcy = p_y0 + (card_y_hi - card_y_lo) / 2
                p_rect = plt.Rectangle(
                    (pcx - 1.0, pcy - 1.0), 2.0, 2.0,
                    facecolor="white", edgecolor="black", linewidth=0.4,
                )
                ax.add_patch(p_rect)

            if t == 0:
                label = row_labels[h_idx] if (row_labels and h_idx < len(row_labels)) else f"head {h_idx}"
                ax.set_ylabel(label, fontsize=8)
            if h_idx == 0:
                ax.set_title(f"t={t}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_per_head_strip(
    per_head_seq: np.ndarray,
    out_path: Path,
    title: str,
    img_h: int = 21,
    img_w: int = 35,
) -> None:
    """One row per attention head; T columns of bilinear-upsampled heatmaps.

    Args:
        per_head_seq: (T, fh, fw, num_heads) per-step per-head attention.
        out_path: PNG path.
        title: figure suptitle.
    """
    T, fh, fw, H = per_head_seq.shape
    fig, axes = plt.subplots(H, T, figsize=(1.6 * T, 1.3 * H + 0.4))
    if H == 1:
        axes = axes[None, :]
    if T == 1:
        axes = axes[:, None]

    vmax = float(per_head_seq.max())

    card_pixel_y_lo = 7
    card_pixel_y_hi = 14

    for h_idx in range(H):
        for t in range(T):
            attn = per_head_seq[t, :, :, h_idx]
            attn_up = jax.image.resize(jnp.asarray(attn), (img_h, img_w), method="bilinear")
            ax = axes[h_idx, t]
            ax.imshow(np.asarray(attn_up), cmap="hot", vmin=0.0, vmax=vmax)
            ax.axhline(y=card_pixel_y_lo - 0.5, color="cyan", lw=0.4, alpha=0.7)
            ax.axhline(y=card_pixel_y_hi - 0.5, color="cyan", lw=0.4, alpha=0.7)
            flat = attn.flatten()
            idx = int(flat.argmax())
            ar, ac = divmod(idx, fw)
            cy = ar * (img_h / fh) + (img_h / fh) / 2
            cx = ac * (img_w / fw) + (img_w / fw) / 2
            ax.text(cx, cy, f"{flat.max():.2f}", ha="center", va="center",
                    color="cyan", fontsize=6, fontweight="bold")
            if t == 0:
                ax.set_ylabel(f"head {h_idx}", fontsize=8)
            if h_idx == 0:
                ax.set_title(f"t={t}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _propagate_card_attention(
    attn_2d_seq: np.ndarray,
    card_row_lo: int,
    card_row_hi: int,
) -> np.ndarray:
    """Build a propagated view: on-card cells carry a running max across time;
    off-card cells show only the current step's value (no propagation).

    Args:
        attn_2d_seq: (T, fh, fw) raw per-step attention maps.
        card_row_lo, card_row_hi: feature-row indices [lo, hi) considered to
            overlap the card row in pixel space.

    Returns:
        (T, fh, fw) propagated attention.
    """
    T, fh, fw = attn_2d_seq.shape
    on_card_mask = np.zeros((fh, fw), dtype=bool)
    on_card_mask[card_row_lo:card_row_hi, :] = True

    propagated = np.zeros_like(attn_2d_seq)
    running_max_on_card = np.zeros((fh, fw), dtype=np.float32)

    for t in range(T):
        cur = attn_2d_seq[t]
        # On-card cells: running max across time
        on_card_now = np.where(on_card_mask, cur, 0.0)
        running_max_on_card = np.maximum(running_max_on_card, on_card_now)
        # Off-card cells: only current step's value (no propagation)
        off_card_now = np.where(on_card_mask, 0.0, cur)
        propagated[t] = running_max_on_card + off_card_now

    return propagated


def _save_attn_strip(
    attn_2d_seq: np.ndarray,
    obs_seq: np.ndarray,
    action_view_slots: list[int],
    out_path: Path,
    title: str,
    agent_idx: int,
    img_h: int = 21,
    img_w: int = 35,
) -> None:
    """Save a 2xT panel: raw numeric grid + blocky attention over dimmed obs."""
    T, fh, fw = attn_2d_seq.shape
    vmax_raw = float(attn_2d_seq.max())
    if vmax_raw < 1e-6:
        vmax_raw = 1.0
    cmap_name = "plasma"
    card_row_lo = 2
    card_row_hi = 4

    fig, axes = plt.subplots(
        2, T, figsize=(2.0 * T + 0.55, 4.0),
        gridspec_kw={"height_ratios": [fh / fw, img_h / img_w]},
    )
    if T == 1:
        axes = axes[:, None]

    label_color = np.asarray(AGENT_0_COLOR if agent_idx == 0 else AGENT_1_COLOR, dtype=np.uint8)
    label_text = "A0" if agent_idx == 0 else "A1"

    for t in range(T):
        attn = attn_2d_seq[t]
        ax_raw = axes[0, t]
        ax = axes[1, t]

        ax_raw.imshow(attn, cmap=cmap_name, vmin=0.0, vmax=vmax_raw,
                      interpolation="nearest", aspect="equal")
        ax_raw.add_patch(
            plt.Rectangle(
                (-0.5, card_row_lo - 0.5),
                fw,
                card_row_hi - card_row_lo,
                fill=False,
                edgecolor="white",
                linewidth=1.1,
            )
        )
        for r in range(fh):
            for c in range(fw):
                val = float(attn[r, c])
                text_color = "black" if val > vmax_raw * 0.55 else "white"
                ax_raw.text(
                    c, r, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=5.5, color=text_color,
                )
        if t == 0:
            ax_raw.set_ylabel("raw", fontsize=8)
        ax_raw.set_title(f"t={t}", fontsize=8)
        ax_raw.set_xticks(range(fw))
        ax_raw.set_yticks(range(fh))
        ax_raw.set_xticklabels([f"c{c}" for c in range(fw)], fontsize=5)
        ax_raw.set_yticklabels([f"r{r}" for r in range(fh)], fontsize=5)
        ax_raw.tick_params(length=0, pad=1)

        base = (np.clip(obs_seq[t], 0.0, 1.0) * 255.0).astype(np.uint8)
        slot = int(action_view_slots[t]) if action_view_slots[t] is not None else -1
        if slot >= 0:
            if t == T - 1:
                base = _draw_card_border_upscaled(base, slot, label_color, thickness=1, scale=1, outset_raw=0)
            else:
                base = _draw_own_action_dot_raw(base, slot, agent_idx)
        base_faded = np.clip(base.astype(np.float32) * 0.45 + 255.0 * 0.55, 0.0, 255.0).astype(np.uint8)
        ax.imshow(base_faded)
        attn_up = jax.image.resize(jnp.asarray(attn), (img_h, img_w), method="nearest")
        attn_up_np = np.asarray(attn_up)
        attn_norm = np.clip(attn_up_np / vmax_raw, 0.0, 1.0)
        rgba = plt.get_cmap(cmap_name)(attn_norm)
        rgba[..., 3] = attn_norm * 0.95
        ax.imshow(rgba, extent=(-0.5, img_w - 0.5, img_h - 0.5, -0.5))
        ax.text(
            1.0,
            img_h - 1.4,
            label_text,
            color=label_color.astype(np.float32) / 255.0,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
        )
        if t == 0:
            ax.set_ylabel("obs", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    sm = plt.cm.ScalarMappable(
        cmap=cmap_name,
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=vmax_raw),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.026, pad=0.012)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.set_label("attention", fontsize=8)

    fig.subplots_adjust(left=0.045, right=0.93, top=0.97, bottom=0.08, wspace=0.08, hspace=0.08)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-idx", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--episode-rng-base", type=int, default=200)
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to <run_dir>/attention_numbers/")
    parser.add_argument("--sampled", action="store_true",
                        help="Sampled actions (default greedy)")
    args = parser.parse_args()

    ev = load_card_game_eval(args.checkpoint)
    greedy = not args.sampled

    run_dir = Path(args.checkpoint).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else (run_dir / "attention_numbers")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-build card masks for view-slot pooling (used in alignment diagnostics).
    # 21x35 image, 6x9 feature grid match the JA image agent's downsampling.
    _card_masks_np = np.asarray(build_card_masks(21, 35, 6, 9))

    # Pass per-head partner-feed config to the rollout so obs is augmented as
    # in training; otherwise scalar_embed param shape doesn't match.
    _policy_scalar_dim = int(getattr(ev.policy.network, "scalar_dim", 0))
    _partner_feed_dim = _policy_scalar_dim if _policy_scalar_dim > 0 else 5
    _use_card_masks = jnp.asarray(_card_masks_np) if _policy_scalar_dim > 0 else None

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  label={ev.label}  seeds={ev.num_seeds}  seed_idx={args.seed_idx}")
    print(f"  max_steps={ev.max_steps}  greedy={greedy}")
    print(f"  output: {out_dir}")

    params = jax.tree.map(lambda x: x[args.seed_idx], ev.params)

    for ep in range(args.num_episodes):
        rng = jax.random.PRNGKey(args.episode_rng_base + args.seed_idx * 1000 + ep)
        ep_states, attn_maps, ep_actions, _ = run_episode_with_states(
            rng, ev.env, params, ev.policy, params, ev.policy, ev.max_steps,
            collect_attention=True, greedy=greedy,
            ja_card_masks=_use_card_masks,
            partner_feed_dim=_partner_feed_dim,
        )

        for agent_key in ("agent_0", "agent_1"):
            png_path = out_dir / f"attn_seed{args.seed_idx}_ep{ep}_{agent_key}.png"

            attn_seq = []
            num_steps = len(attn_maps[agent_key])
            for t in range(num_steps):
                raw = np.asarray(attn_maps[agent_key][t]).squeeze()
                if raw.ndim == 3:
                    attn_2d = raw.mean(axis=-1)
                elif raw.ndim == 2:
                    attn_2d = raw
                else:
                    attn_2d = raw.reshape(6, 9)
                attn_seq.append(attn_2d)

            attn_seq_np = np.stack(attn_seq, axis=0)

            obs_seq = []
            for t in range(len(ep_states)):
                obs_seq.append(_render_agent_view(ep_states[t], agent_key, t))
            obs_seq_np = np.stack(obs_seq, axis=0)
            T_attn = attn_seq_np.shape[0]
            T_obs = obs_seq_np.shape[0]
            if T_obs != T_attn:
                obs_seq_np = obs_seq_np[:T_attn] if T_obs > T_attn else np.concatenate(
                    [obs_seq_np, np.zeros((T_attn - T_obs, 21, 35, 3), dtype=np.float32)],
                    axis=0,
                )

            agent_idx = 0 if agent_key == "agent_0" else 1
            action_slots = [int(a[agent_idx]) for a in ep_actions[:T_attn]]
            _save_attn_strip(
                attn_seq_np,
                obs_seq_np,
                action_slots,
                png_path,
                title=f"seed {args.seed_idx} ep {ep} {agent_key}",
                agent_idx=agent_idx,
            )
            print(f"[saved {png_path}]")


if __name__ == "__main__":
    main()
