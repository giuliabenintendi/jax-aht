"""Visualize the full pipeline that turns raw 6x9 spatial attention into a
5-dim canonical-frame attention-on-card vector — the quantity used by JA
attention shaping rewards.

Pipeline (per step, per agent):
    raw_attn  shape (fh=6, fw=9)            — from the model
       |
       |  multiply elementwise by per-card overlap masks (5, fh, fw)
       v
    per_card_attn  shape (5,)              — view-frame: how much attention on each view-slot card
       |
       |  scatter via OP position permutation
       v
    canonical_card_attn  shape (5,)        — GT-frame: how much attention on each GT card identity

Run on a trained checkpoint and emit one PNG per (seed, episode, agent, step).

Usage:
    ./run_gpu.sh 5 evaluation.card_attn_pipeline \\
        --checkpoint <path>/saved_train_run \\
        --seed-idx 0 --episode 0 --step 4 --agent agent_0
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

from agents.ja_utils import build_card_masks
from envs.card_game.rendering import (
    CARD_COLORS, NUM_CARDS, TILE_PIXELS, render_card_game_minimal,
)
from evaluation._card_game_utils import load_card_game_eval
from evaluation.vis_episodes import run_episode_with_states


IMG_H, IMG_W = 21, 35
FEAT_H, FEAT_W = 6, 9


def _walk_to_card_state(state):
    s = state
    while hasattr(s, "env_state") and not hasattr(s, "card_permutation"):
        s = s.env_state
    return s


def _render_agent_view(state, agent_key: str, step_count: int) -> np.ndarray:
    """Same as in attention_numbers.py — reconstruct agent's own-frame obs."""
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


def _save_pipeline_figure(
    obs: np.ndarray,
    attn_2d: np.ndarray,
    card_masks: np.ndarray,
    pos_perm: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Emit a single figure showing every step of the card-attention pipeline.

    Args:
        obs: (img_h, img_w, 3) agent-frame obs.
        attn_2d: (fh, fw) attention map (probably head-averaged).
        card_masks: (5, fh, fw) per-card overlap masks.
        pos_perm: (5,) the agent's position permutation.
        out_path: PNG path.
        title: figure suptitle.
    """
    H, W = obs.shape[:2]
    fh, fw = attn_2d.shape

    # Per-card products and sums
    products = attn_2d[None, :, :] * card_masks   # (5, fh, fw)
    view_card_attn = products.sum(axis=(1, 2))    # (5,) view-frame

    # Scatter to canonical frame: canonical[pos_perm[k]] = view_card_attn[k]
    canonical_card_attn = np.zeros(NUM_CARDS, dtype=np.float32)
    canonical_card_attn[pos_perm] = view_card_attn

    card_colors_np = np.asarray(CARD_COLORS) / 255.0

    # Layout: 4 rows
    #   row 0: obs | obs + raw attention overlay | raw attention grid
    #   row 1: 5 card masks
    #   row 2: 5 elementwise products (attn * mask[c])
    #   row 3: 2 wide panels — view-frame bar chart | canonical-frame bar chart
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(4, 5, height_ratios=[1.1, 1.0, 1.0, 1.0],
                           hspace=0.55, wspace=0.25)

    # ----- Row 0: obs, obs+attention overlay, raw attention grid -----
    ax_obs = fig.add_subplot(gs[0, 0])
    ax_obs.imshow(obs)
    ax_obs.set_title("agent obs (own frame)", fontsize=10)
    ax_obs.axis("off")

    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_overlay.imshow(obs)
    attn_up = np.asarray(jax.image.resize(
        jnp.asarray(attn_2d), (H, W), method="bilinear"
    ))
    ax_overlay.imshow(attn_up, cmap="hot", alpha=0.55,
                      vmin=0.0, vmax=float(attn_2d.max()),
                      extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    ax_overlay.axhline(y=7 - 0.5, color="cyan", lw=0.4, alpha=0.7)
    ax_overlay.axhline(y=14 - 0.5, color="cyan", lw=0.4, alpha=0.7)
    ax_overlay.set_title("obs + attention overlay\n(cyan = card-row band)",
                          fontsize=10)
    ax_overlay.axis("off")

    ax_attn = fig.add_subplot(gs[0, 2])
    ax_attn.imshow(attn_2d, cmap="hot", vmin=0.0, vmax=float(attn_2d.max()),
                   interpolation="nearest", aspect="equal")
    for r in range(fh):
        for c in range(fw):
            v = attn_2d[r, c]
            tc = "black" if v > attn_2d.max() * 0.6 else "white"
            ax_attn.text(c, r, f"{v:.2f}", ha="center", va="center",
                         fontsize=5.5, color=tc)
    ax_attn.set_title(f"raw attention ({fh}x{fw})", fontsize=10)
    ax_attn.set_xticks(range(fw))
    ax_attn.set_yticks(range(fh))
    ax_attn.set_xticklabels([f"c{c}" for c in range(fw)], fontsize=5)
    ax_attn.set_yticklabels([f"r{r}" for r in range(fh)], fontsize=5)
    ax_attn.tick_params(length=0, pad=1)

    # Annotation column on the right of row 0 explaining the next steps
    ax_note = fig.add_subplot(gs[0, 3:])
    ax_note.axis("off")
    ax_note.text(0.0, 0.95,
        "How attention-on-card is computed:\n\n"
        "1. For each card c in [0..4]:\n"
        "     elementwise:  product[c] = raw_attn * card_mask[c]\n"
        "     sum over (h, w):  view_card_attn[c] = product[c].sum()\n\n"
        "2. Translate to GT (canonical) frame:\n"
        "     canonical[pos_perm[k]] = view_card_attn[k]\n\n"
        "Equivalent einsum:\n"
        "     view_card_attn = einsum('hw,chw->c', attn, card_masks)\n\n"
        "JA shaping uses canonical_card_attn (both agents in the same frame).",
        fontsize=10, family="monospace", va="top")

    # ----- Row 1: 5 card masks -----
    for c in range(NUM_CARDS):
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(card_masks[c], cmap="Blues", vmin=0.0, vmax=1.0,
                  interpolation="nearest", aspect="equal")
        for r in range(fh):
            for cc in range(fw):
                v = card_masks[c, r, cc]
                if v > 0.01:
                    tc = "white" if v > 0.6 else "black"
                    ax.text(cc, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=5, color=tc)
        # title shows view-slot index and the colour of that view-slot's card
        slot_color = card_colors_np[pos_perm[c]]
        ax.set_title(f"card_mask[c={c}]\nview-slot {c}, GT={int(pos_perm[c])}",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        # colored border to indicate the card's displayed colour
        for spine in ax.spines.values():
            spine.set_edgecolor(slot_color)
            spine.set_linewidth(2.0)
            spine.set_visible(True)

    # ----- Row 2: products attn * card_mask[c] -----
    prod_vmax = float(products.max())
    for c in range(NUM_CARDS):
        ax = fig.add_subplot(gs[2, c])
        ax.imshow(products[c], cmap="hot", vmin=0.0, vmax=prod_vmax,
                  interpolation="nearest", aspect="equal")
        for r in range(fh):
            for cc in range(fw):
                v = products[c, r, cc]
                if v > 0.005:
                    tc = "black" if v > prod_vmax * 0.6 else "white"
                    ax.text(cc, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=5, color=tc)
        ax.set_title(f"attn * card_mask[{c}]\nsum = {view_card_attn[c]:.3f}",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # ----- Row 3: bar charts (view-frame and canonical-frame) -----
    ax_view = fig.add_subplot(gs[3, 0:2])
    bar_colors_view = [card_colors_np[pos_perm[c]] for c in range(NUM_CARDS)]
    ax_view.bar(np.arange(NUM_CARDS), view_card_attn, color=bar_colors_view,
                edgecolor="black", linewidth=0.4)
    ax_view.set_xticks(np.arange(NUM_CARDS))
    ax_view.set_xticklabels([f"slot {c}\n(GT {int(pos_perm[c])})"
                              for c in range(NUM_CARDS)], fontsize=8)
    ax_view.set_ylabel("attention mass", fontsize=9)
    ax_view.set_title("view-frame card attention\n"
                       "(per view-slot in agent's image)", fontsize=10)
    ax_view.set_ylim(0, max(view_card_attn.max(), 0.01) * 1.15)
    ax_view.grid(axis="y", alpha=0.3)

    ax_can = fig.add_subplot(gs[3, 2:4])
    bar_colors_can = [card_colors_np[c] for c in range(NUM_CARDS)]
    ax_can.bar(np.arange(NUM_CARDS), canonical_card_attn, color=bar_colors_can,
               edgecolor="black", linewidth=0.4)
    ax_can.set_xticks(np.arange(NUM_CARDS))
    ax_can.set_xticklabels([f"GT card {c}" for c in range(NUM_CARDS)],
                            fontsize=8)
    ax_can.set_ylabel("attention mass", fontsize=9)
    ax_can.set_title("canonical-frame card attention\n"
                      "(after scatter by pos_perm — used by JA shaping)",
                      fontsize=10)
    ax_can.set_ylim(0, max(canonical_card_attn.max(), 0.01) * 1.15)
    ax_can.grid(axis="y", alpha=0.3)

    # Position-permutation summary on the right
    ax_perm = fig.add_subplot(gs[3, 4])
    ax_perm.axis("off")
    perm_text = "pos_perm (this agent's view):\n\n"
    for k in range(NUM_CARDS):
        perm_text += f"  view_slot {k}  ->  GT card {int(pos_perm[k])}\n"
    ax_perm.text(0.0, 0.95, perm_text, fontsize=9, family="monospace",
                 va="top")

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-idx", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step", type=int, default=4,
                        help="Which step of the episode to visualise.")
    parser.add_argument("--agent", default="agent_0",
                        choices=["agent_0", "agent_1"])
    parser.add_argument("--head", type=int, default=-1,
                        help="Which attention head to use (-1 = head-averaged, the default for JA shaping).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--episode-rng-base", type=int, default=200)
    parser.add_argument("--sampled", action="store_true")
    args = parser.parse_args()

    ev = load_card_game_eval(args.checkpoint)
    greedy = not args.sampled

    run_dir = Path(args.checkpoint).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else (run_dir / "card_attn_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  label={ev.label}  seed={args.seed_idx}  ep={args.episode}  step={args.step}  agent={args.agent}")
    print(f"  head={'avg' if args.head < 0 else args.head}  greedy={greedy}")

    params = jax.tree.map(lambda x: x[args.seed_idx], ev.params)
    rng = jax.random.PRNGKey(args.episode_rng_base + args.seed_idx * 1000 + args.episode)
    ep_states, attn_maps, ep_actions, ep_messages = run_episode_with_states(
        rng, ev.env, params, ev.policy, params, ev.policy, ev.max_steps,
        collect_attention=True, greedy=greedy,
    )

    t = args.step
    state_t = ep_states[t]
    obs = _render_agent_view(state_t, args.agent, step_count=t)

    raw = np.asarray(attn_maps[args.agent][t]).squeeze()
    if raw.ndim == 3 and args.head >= 0:
        attn_2d = raw[..., args.head]
        head_label = f"head {args.head}"
    elif raw.ndim == 3:
        attn_2d = raw.mean(axis=-1)
        head_label = "head-averaged"
    else:
        attn_2d = raw
        head_label = "single head"

    pos_perm = np.asarray(state_t.env_state.per_agent_perm[args.agent])
    card_masks = np.asarray(build_card_masks(IMG_H, IMG_W, FEAT_H, FEAT_W))

    out_path = out_dir / (
        f"card_attn_pipeline_seed{args.seed_idx}_ep{args.episode}_t{t}_"
        f"{args.agent}_{'avg' if args.head < 0 else f'h{args.head}'}.png"
    )
    _save_pipeline_figure(
        obs, attn_2d, card_masks, pos_perm, out_path,
        title=f"attention-on-card pipeline   |   seed {args.seed_idx}  "
              f"ep {args.episode}  t={t}  {args.agent}  ({head_label})",
    )
    print(f"\n[saved {out_path}]")


if __name__ == "__main__":
    main()
