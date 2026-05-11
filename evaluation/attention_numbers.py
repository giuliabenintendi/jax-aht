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

from evaluation._card_game_utils import load_card_game_eval
from evaluation.vis_episodes import run_episode_with_states


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


def _save_attn_strip(
    attn_2d_seq: np.ndarray,
    entropies: list,
    out_path: Path,
    title: str,
    img_h: int = 21,
    img_w: int = 35,
) -> None:
    """Save a 1xT strip of per-step attention heatmaps.

    Each panel shows the (fh, fw) attention map bilinearly upsampled to
    (img_h, img_w). The card row in pixel space (y=7..13) is marked with
    horizontal lines so you can see which cells fall on cards vs background.

    Args:
        attn_2d_seq: (T, fh, fw) attention maps.
        entropies: list of per-step entropy values (nats).
        out_path: PNG path.
        title: figure suptitle.
        img_h, img_w: target heatmap pixel size (the obs resolution).
    """
    T, fh, fw = attn_2d_seq.shape
    fig, axes = plt.subplots(1, T, figsize=(1.7 * T, 2.0))
    if T == 1:
        axes = [axes]

    vmax = float(attn_2d_seq.max())

    for t in range(T):
        attn = attn_2d_seq[t]
        attn_up = jax.image.resize(jnp.asarray(attn), (img_h, img_w), method="bilinear")
        ax = axes[t]
        im = ax.imshow(np.asarray(attn_up), cmap="hot", vmin=0.0, vmax=vmax)

        # Mark card row (pixel y=7..13). Two yellow horizontal lines.
        ax.axhline(y=7 - 0.5, color="yellow", lw=0.6, alpha=0.9)
        ax.axhline(y=14 - 0.5, color="yellow", lw=0.6, alpha=0.9)
        # Mark card column boundaries (each card spans 7 px starting at x=0)
        for c in range(1, 5):
            ax.axvline(x=c * 7 - 0.5, color="yellow", lw=0.3, alpha=0.6)

        # Annotate argmax cell with its value
        flat = attn.flatten()
        idx = int(flat.argmax())
        ar, ac = divmod(idx, fw)
        cy = ar * (img_h / fh) + (img_h / fh) / 2
        cx = ac * (img_w / fw) + (img_w / fw) / 2
        ax.text(
            cx, cy, f"{flat.max():.2f}",
            ha="center", va="center",
            color="cyan", fontsize=8, fontweight="bold",
        )

        ax.set_title(f"t={t}\nH={entropies[t]:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
        )

        for agent_key in ("agent_0", "agent_1"):
            print(f"\n=== seed {args.seed_idx} ep {ep} {agent_key} ===")
            csv_path = out_dir / f"attn_seed{args.seed_idx}_ep{ep}_{agent_key}.csv"
            png_path = out_dir / f"attn_seed{args.seed_idx}_ep{ep}_{agent_key}.png"

            attn_seq = []
            entropies = []

            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "row", "col", "attn"])

                num_steps = len(attn_maps[agent_key])
                for t in range(num_steps):
                    attn_2d = np.asarray(attn_maps[agent_key][t]).squeeze()
                    if attn_2d.ndim != 2:
                        attn_2d = attn_2d.reshape(6, 9)
                    flat = attn_2d.flatten()
                    flat_sum = float(flat.sum())
                    ent = _entropy(flat / max(flat_sum, 1e-12))
                    argmax_flat = int(flat.argmax())
                    argmax_r, argmax_c = divmod(argmax_flat, attn_2d.shape[1])
                    print(f"\nstep={t}  sum={flat_sum:.4f}  entropy={ent:.3f} nats  "
                          f"argmax=(r={argmax_r}, c={argmax_c})  max_val={flat.max():.4f}")
                    print(_format_attn_grid(attn_2d))

                    attn_seq.append(attn_2d)
                    entropies.append(ent)
                    for r in range(attn_2d.shape[0]):
                        for c in range(attn_2d.shape[1]):
                            w.writerow([t, r, c, f"{attn_2d[r, c]:.6f}"])
            print(f"[saved {csv_path}]")

            attn_seq_np = np.stack(attn_seq, axis=0)
            _save_attn_strip(
                attn_seq_np,
                entropies,
                png_path,
                title=f"seed {args.seed_idx} ep {ep} {agent_key} "
                      f"(yellow lines = card row boundaries y=7,14)",
            )
            print(f"[saved {png_path}]")


if __name__ == "__main__":
    main()
