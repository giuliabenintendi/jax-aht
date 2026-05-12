"""Dump per-tile attention values from a trained JA checkpoint, as numbers.

For each of N self-play episodes, run the policy and print the 6x9 attention
map at every step as a numerical table. Useful to see whether the trained
policy actually concentrates attention on a small number of cells or spreads
it across the grid.

Outputs:
  - stdout: tables per step, plus per-step argmax cell and entropy
  - <out>/attn_seed{i}_ep{j}.csv: flat (step, row, col, value) rows

Usage:
    ./run_gpu.sh 5 evaluation.attention_numbers \\
        --checkpoint /scratch/.../saved_train_run \\
        --seed-idx 0 \\
        --num-episodes 2
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.card_game.rendering import (
    CARD_COLORS, NUM_CARDS, TILE_PIXELS, render_card_game_minimal,
)
from agents.ja_utils import build_card_masks
from evaluation._card_game_utils import load_card_game_eval
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
    step's messages; the cyan card-row lines in the final plot are enough
    to read where the heat is falling).
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


def _format_attn_grid(attn_2d: np.ndarray, decimals: int = 3) -> str:
    """Render a 2D attention grid as an aligned text table."""
    h, w = attn_2d.shape
    width = decimals + 3  # leading "0." + decimals + space
    rows = []
    rows.append("     " + "  ".join(f"c{c:<{width-2}}" for c in range(w)))
    for r in range(h):
        cells = "  ".join(f"{attn_2d[r, c]:.{decimals}f}" for c in range(w))
        rows.append(f"r{r}: {cells}")
    return "\n".join(rows)


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


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
) -> None:
    """4 rows (one per head) × T columns; obs as background, head-specific
    attention overlay, and a colored marker on the card the agent acted on
    (dot for deliberation message, box for decision pick).

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
    vmax = float(per_head_seq.max())
    TP = 7  # TILE_PIXELS
    card_y_lo, card_y_hi = 7, 14

    for h_idx in range(H):
        for t in range(T):
            ax = axes[h_idx, t]
            ax.imshow(obs_seq[t])
            attn = per_head_seq[t, :, :, h_idx]
            attn_up = jax.image.resize(jnp.asarray(attn), (img_h, img_w), method="bilinear")
            ax.imshow(np.asarray(attn_up), cmap="hot", alpha=0.55,
                      vmin=0.0, vmax=vmax,
                      extent=(-0.5, img_w - 0.5, img_h - 0.5, -0.5))

            slot = action_view_slots[t]
            if slot is not None and slot >= 0:
                x0 = slot * TP - 0.5
                y0 = card_y_lo - 0.5
                if is_decision_seq[t]:
                    rect = plt.Rectangle(
                        (x0, y0), TP, card_y_hi - card_y_lo,
                        fill=False, edgecolor=marker_color, linewidth=2.0,
                    )
                    ax.add_patch(rect)
                else:
                    cx = x0 + TP / 2
                    cy = y0 + (card_y_hi - card_y_lo) / 2
                    circ = plt.Circle(
                        (cx, cy), radius=1.5,
                        facecolor=marker_color, edgecolor="black", linewidth=0.4,
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
                ax.set_ylabel(f"head {h_idx}", fontsize=8)
            if h_idx == 0:
                ax.set_title(f"t={t}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title + "  (coloured dot/box = own action; white square = partner msg)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
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

    fig.suptitle(title + "  (per-head attention, cyan = card-row boundaries)", fontsize=10)
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
    entropies: list,
    obs_seq: np.ndarray,
    out_path: Path,
    title: str,
    img_h: int = 21,
    img_w: int = 35,
) -> None:
    """Save a 3xT panel:
      row 0 = raw 6x9 grid with values
      row 1 = bilinear-upsampled attention overlaid on the agent's own-frame obs
      row 2 = propagated attention (on-card running max, off-card current only)
              overlaid on the obs

    Args:
        attn_2d_seq: (T, fh, fw) attention maps.
        entropies: list of per-step entropy values (nats).
        obs_seq: (T, img_h, img_w, 3) per-step agent-view images, [0, 1] float.
        out_path: PNG path.
        title: figure suptitle.
    """
    T, fh, fw = attn_2d_seq.shape

    card_pixel_y_lo = 7
    card_pixel_y_hi = 14
    # Feature rows fully or mostly overlapping the card pixel row.
    # With fh=6, img_h=21: r=2 covers y=7-10.5, r=3 covers y=10.5-14. Both on card.
    card_feat_r_lo = int(round(card_pixel_y_lo * fh / img_h))   # 2
    card_feat_r_hi = int(round(card_pixel_y_hi * fh / img_h))   # 4

    propagated = _propagate_card_attention(
        attn_2d_seq, card_feat_r_lo, card_feat_r_hi,
    )

    vmax_raw = float(attn_2d_seq.max())
    vmax_prop = float(propagated.max())

    fig, axes = plt.subplots(3, T, figsize=(2.2 * T, 6.0),
                              gridspec_kw={"height_ratios": [fh / fw, img_h / img_w, img_h / img_w]})
    if T == 1:
        axes = axes[:, None]

    for t in range(T):
        attn = attn_2d_seq[t]
        prop = propagated[t]

        # --- Row 0: raw 6x9 grid, value-annotated ---
        ax = axes[0, t]
        ax.imshow(attn, cmap="hot", vmin=0.0, vmax=vmax_raw,
                  interpolation="nearest", aspect="equal")
        for r in range(fh):
            for c in range(fw):
                val = attn[r, c]
                text_color = "black" if val > vmax_raw * 0.6 else "white"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color=text_color)
        ax.axhline(y=card_feat_r_lo - 0.5, color="cyan", lw=0.8, alpha=0.9)
        ax.axhline(y=card_feat_r_hi - 0.5, color="cyan", lw=0.8, alpha=0.9)
        ax.set_title(f"t={t}\nH={entropies[t]:.2f} nats", fontsize=8)
        ax.set_xticks(range(fw))
        ax.set_yticks(range(fh))
        ax.set_xticklabels([f"c{c}" for c in range(fw)], fontsize=5)
        ax.set_yticklabels([f"r{r}" for r in range(fh)], fontsize=5)
        ax.tick_params(length=0, pad=1)

        # --- Row 1: obs + raw attention overlay ---
        ax2 = axes[1, t]
        ax2.imshow(obs_seq[t])
        attn_up = jax.image.resize(jnp.asarray(attn), (img_h, img_w), method="bilinear")
        ax2.imshow(np.asarray(attn_up), cmap="hot", alpha=0.55,
                   vmin=0.0, vmax=vmax_raw,
                   extent=(-0.5, img_w - 0.5, img_h - 0.5, -0.5))
        ax2.axhline(y=card_pixel_y_lo - 0.5, color="cyan", lw=0.4, alpha=0.7)
        ax2.axhline(y=card_pixel_y_hi - 0.5, color="cyan", lw=0.4, alpha=0.7)
        if t == 0:
            ax2.set_ylabel("raw", fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # --- Row 2: obs + propagated attention overlay ---
        ax3 = axes[2, t]
        ax3.imshow(obs_seq[t])
        prop_up = jax.image.resize(jnp.asarray(prop), (img_h, img_w), method="bilinear")
        ax3.imshow(np.asarray(prop_up), cmap="hot", alpha=0.55,
                   vmin=0.0, vmax=vmax_prop,
                   extent=(-0.5, img_w - 0.5, img_h - 0.5, -0.5))
        ax3.axhline(y=card_pixel_y_lo - 0.5, color="cyan", lw=0.4, alpha=0.7)
        ax3.axhline(y=card_pixel_y_hi - 0.5, color="cyan", lw=0.4, alpha=0.7)
        if t == 0:
            ax3.set_ylabel("propagated", fontsize=8)
        ax3.set_xticks([])
        ax3.set_yticks([])

    fig.suptitle(title + "  (cyan lines = card-row boundaries; "
                         "row 2 carries on-card max forward)", fontsize=10)
    fig.tight_layout()
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
        ep_states, attn_maps, ep_actions, ep_messages = run_episode_with_states(
            rng, ev.env, params, ev.policy, params, ev.policy, ev.max_steps,
            collect_attention=True, greedy=greedy,
            ja_card_masks=_use_card_masks,
            partner_feed_dim=_partner_feed_dim,
        )

        for agent_key in ("agent_0", "agent_1"):
            print(f"\n=== seed {args.seed_idx} ep {ep} {agent_key} ===")
            csv_path = out_dir / f"attn_seed{args.seed_idx}_ep{ep}_{agent_key}.csv"
            png_path = out_dir / f"attn_seed{args.seed_idx}_ep{ep}_{agent_key}.png"

            attn_seq = []
            entropies = []
            attn_per_head_seq = []  # (T, fh, fw, num_heads) if available

            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "row", "col", "attn"])

                num_steps = len(attn_maps[agent_key])
                for t in range(num_steps):
                    raw = np.asarray(attn_maps[agent_key][t]).squeeze()
                    # raw may be (fh, fw) (old) or (fh, fw, num_heads) (new)
                    if raw.ndim == 3:
                        attn_per_head_seq.append(raw)
                        attn_2d = raw.mean(axis=-1)
                    elif raw.ndim == 2:
                        attn_2d = raw
                    else:
                        attn_2d = raw.reshape(6, 9)
                    flat = attn_2d.flatten()
                    flat_sum = float(flat.sum())
                    ent = _entropy(flat / max(flat_sum, 1e-12))
                    argmax_flat = int(flat.argmax())
                    argmax_r, argmax_c = divmod(argmax_flat, attn_2d.shape[1])
                    print(f"\nstep={t}  sum={flat_sum:.4f}  entropy={ent:.3f} nats  "
                          f"argmax=(r={argmax_r}, c={argmax_c})  max_val={flat.max():.4f}")
                    print(_format_attn_grid(attn_2d))

                    # Per-head breakdown (when available)
                    if raw.ndim == 3:
                        num_heads = raw.shape[-1]
                        for h_idx in range(num_heads):
                            head_attn = raw[..., h_idx]
                            head_flat = head_attn.flatten()
                            head_sum = float(head_flat.sum())
                            head_ent = _entropy(head_flat / max(head_sum, 1e-12))
                            head_argmax = int(head_flat.argmax())
                            har, hac = divmod(head_argmax, head_attn.shape[1])
                            print(f"  head {h_idx}  sum={head_sum:.4f}  entropy={head_ent:.3f}  "
                                  f"argmax=(r={har}, c={hac})  max_val={head_flat.max():.4f}")
                            print("  " + _format_attn_grid(head_attn).replace("\n", "\n  "))

                    attn_seq.append(attn_2d)
                    entropies.append(ent)
                    for r in range(attn_2d.shape[0]):
                        for c in range(attn_2d.shape[1]):
                            w.writerow([t, r, c, f"{attn_2d[r, c]:.6f}"])
            print(f"[saved {csv_path}]")

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

            _save_attn_strip(
                attn_seq_np,
                entropies,
                obs_seq_np,
                png_path,
                title=f"seed {args.seed_idx} ep {ep} {agent_key}",
            )
            print(f"[saved {png_path}]")

            if attn_per_head_seq:
                per_head_seq_np = np.stack(attn_per_head_seq, axis=0)
                if per_head_seq_np.shape[0] != T_attn:
                    per_head_seq_np = per_head_seq_np[:T_attn]
                per_head_path = out_dir / (
                    f"attn_seed{args.seed_idx}_ep{ep}_{agent_key}_per_head.png"
                )
                _save_per_head_strip(
                    per_head_seq_np,
                    per_head_path,
                    title=f"seed {args.seed_idx} ep {ep} {agent_key}",
                )
                print(f"[saved {per_head_path}]")

                # Per-head with obs and action overlay.
                agent_idx_int = int(agent_key.split("_")[1])
                partner_idx_int = 1 - agent_idx_int
                action_view_slots = []
                is_decision_seq = []
                partner_msg_view_slots = []
                for t in range(T_attn):
                    state_t = ep_states[t]
                    pos_perm = np.asarray(state_t.env_state.per_agent_perm[agent_key])
                    inv_recol = np.asarray(state_t.per_agent_inv_recolouring[agent_key])
                    pos_perm_inv = np.argsort(pos_perm)

                    # ep_actions has picks (-1 on deliberation); ep_messages
                    # has messages (-1 at decision). Use whichever applies.
                    is_decision_t = (t == T_attn - 1)
                    if is_decision_t:
                        action_t = int(ep_actions[t][agent_idx_int])
                    else:
                        action_t = (
                            int(ep_messages[t][agent_idx_int])
                            if len(ep_messages) > t else -1
                        )
                    if action_t >= 0:
                        action_gt = int(inv_recol[action_t])
                        action_view_slots.append(int(pos_perm_inv[action_gt]))
                    else:
                        action_view_slots.append(-1)
                    is_decision_seq.append(is_decision_t)

                    # Partner's last-emitted GT message (env_state.messages is GT frame).
                    # Walk through wrappers to find the inner CardGameState.
                    inner = state_t
                    while hasattr(inner, "env_state") and not hasattr(inner, "messages"):
                        inner = inner.env_state
                    if hasattr(inner, "messages"):
                        partner_msg_gt = int(np.asarray(inner.messages)[partner_idx_int])
                        if partner_msg_gt >= 0:
                            partner_msg_view_slots.append(int(pos_perm_inv[partner_msg_gt]))
                        else:
                            partner_msg_view_slots.append(-1)
                    else:
                        partner_msg_view_slots.append(-1)

                action_overlay_path = out_dir / (
                    f"attn_seed{args.seed_idx}_ep{ep}_{agent_key}_per_head_action.png"
                )
                _save_per_head_action_overlay(
                    per_head_seq_np,
                    obs_seq_np[:T_attn],
                    action_view_slots,
                    is_decision_seq,
                    partner_msg_view_slots,
                    action_overlay_path,
                    title=f"seed {args.seed_idx} ep {ep} {agent_key}",
                    agent_idx=agent_idx_int,
                )
                print(f"[saved {action_overlay_path}]")


if __name__ == "__main__":
    main()
