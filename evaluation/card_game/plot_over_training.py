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

# (display label, csv filename, color) — matches the bar plot palette
CONDS = [
    ("OP only",                "op_only.csv",          "#2E7DF0"),  # blue
    ("OP + JA + shaping",      "op_ja_shaping.csv",    "#51B18D"),  # green
    ("OP + comm + shaping",    "op_comm_shaping.csv",  "#F55F74"),  # Abet 823
]


def main() -> None:
    fig, (ax_sp, ax_xp) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for label, fname, color in CONDS:
        path = CSV_DIR / fname
        if not path.exists():
            print(f"skip: {label}  (no {path})")
            continue
        df = pd.read_csv(path).sort_values("env_step").reset_index(drop=True)
        x = df["env_step"].values
        sp_m, sp_s = df["sp_mean"].values, df["sp_std"].values
        xp_m, xp_s = df["xp_mean"].values, df["xp_std"].values

        ax_sp.plot(x, sp_m, color=color, lw=2.4, label=label, zorder=4)
        ax_sp.fill_between(x, sp_m - sp_s, sp_m + sp_s,
                           color=color, alpha=0.20, zorder=3)
        ax_xp.plot(x, xp_m, color=color, lw=2.4, label=label, zorder=4)
        ax_xp.fill_between(x, xp_m - xp_s, xp_m + xp_s,
                           color=color, alpha=0.20, zorder=3)
        print(f"{label:<24s} final SP={sp_m[-1]:.3f}  XP={xp_m[-1]:.3f}  "
              f"(n_chunks={len(x)})")

    for ax, title in [(ax_sp, "Training"), (ax_xp, "Testing (Zero-Shot)")]:
        ax.axhline(RANDOM_BASELINE, ls="--", lw=1.5, color=RANDOM_COLOR, zorder=2)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Environment steps", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=12)

    ax_sp.set_ylabel("Episode return", fontsize=14)
    ax_xp.legend(frameon=False, fontsize=12, loc="lower right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
