"""Two-panel SP/XP curves over training (Hu et al. 2021 "Other-Play" style).

Left: in-training self-play return at each saved checkpoint.
Right: cross-play (zero-shot) return at the same checkpoint, paired across seeds.
Both panels share the y-axis. One line per condition with ±1 std shaded band.

Reads aggregated CSVs produced by eval_over_training.py
  (one CSV per condition, columns: condition, chunk, env_step,
   sp_mean, sp_std, xp_mean, xp_std).

Usage:
    uv run python evaluation/card_game/plot_over_training.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_DIR = Path(__file__).parent / "over_training"
OUT = Path("plots/card_game/over_training.png")

RANDOM_BASELINE = 0.20
RANDOM_COLOR = "#d62728"

# Per-chunk eval subsamples to this many seeds (see run_over_training_eval.sh).
# Used to convert the CSV's std columns into SEM for the shaded bands.
N_SEEDS = 10

# (display label, csv filename, color)
# teal = No-OP baseline (collapses in XP — private color conventions);
# pink/blue = OP-trained methods (coordination should transfer).
CONDS = [
    ("No OP",                  "self_play.csv",        "#0d7d87"),  # teal
    ("OP + JA + shaping",      "op_ja_shaping.csv",    "#f77f74"),  # pink
    ("OP + comm + shaping",    "op_comm_shaping.csv",  "#8cc5e3"),  # blue
]


def main() -> None:
    fig, (ax_sp, ax_xp) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for label, fname, color in CONDS:
        path = CSV_DIR / fname
        if not path.exists():
            print(f"skip: {label}  (no {path})")
            continue
        df = pd.read_csv(path).sort_values("env_step").reset_index(drop=True)
        # Normalize x to fraction of this condition's training (0 -> 1) so
        # the 5M and 15M runs are visually comparable end-to-end.
        env_step = df["env_step"].values
        x = env_step / env_step.max()
        sp_m = df["sp_mean"].values
        xp_m = df["xp_mean"].values
        # CSV stores std; convert to SEM across the N_SEEDS evaluated per chunk.
        sp_e = df["sp_std"].values / np.sqrt(N_SEEDS)
        xp_e = df["xp_std"].values / np.sqrt(N_SEEDS)

        ax_sp.plot(x, sp_m, color=color, lw=2.4, label=label, zorder=4)
        ax_sp.fill_between(x, sp_m - sp_e, sp_m + sp_e,
                           color=color, alpha=0.20, zorder=3)
        ax_xp.plot(x, xp_m, color=color, lw=2.4, label=label, zorder=4)
        ax_xp.fill_between(x, xp_m - xp_e, xp_m + xp_e,
                           color=color, alpha=0.20, zorder=3)
        print(f"{label:<24s} final SP={sp_m[-1]:.3f}  XP={xp_m[-1]:.3f}  "
              f"(n_chunks={len(x)}, max env_step={int(env_step.max()):,})")

    for ax, title in [(ax_sp, "Training"), (ax_xp, "Testing (Zero-Shot)")]:
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Training progress", fontsize=14)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=12)

    ax_sp.set_ylabel("Episode return", fontsize=14)

    handles, labels = ax_sp.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=12,
               loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
