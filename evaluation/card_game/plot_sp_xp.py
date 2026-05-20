"""Grouped SP/XP bar plot for the card-game results table.

One base colour per condition; SP = solid fill, XP = white column with a
diagonal hatch in the same colour. Dashed random-baseline line on top,
legend top-left.

Bars and error bars are computed from each run's cross-play matrix
(`xp_score_matrix.csv` in `xp_matrices/`), so SP and XP come from one eval:
  SP height = mean of the matrix diagonal (seed i paired with itself)
  SP error  = std(diagonal, ddof=1) / sqrt(N)        -- N independent seeds
  XP height = mean of every off-diagonal cell (all cross-play pairs)
  XP error  = delete-one-seed jackknife SE of that all-pairs mean (see xp_stats.py)

Usage:
    uv run --no-project --with matplotlib --with numpy --with scipy \
        python evaluation/card_game/plot_sp_xp.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from xp_stats import jackknife_xp, parse_xp_matrix

MATRIX_DIR = Path(__file__).parent / "xp_matrices"
OUT = Path("plots/card_game/sp_xp_bar.png")

RANDOM_COLOR = "#d62728"
RANDOM_BASELINE = 0.20
XP_HATCH = "//"

# (display label, csv filename, base colour)
ROWS = [
    ("OP only", "op_only.csv", "#2E7DF0"),                   # electric blue
    ("OP + JA", "op_ja.csv", "#F5871F"),                     # orange
    ("OP + JA\n+ shaping", "op_ja_shaping.csv", "#51B18D"),  # green
    ("OP + comm", "op_comm.csv", "#8C82F6"),                 # purple
]


def sp_mean_sem(mat: np.ndarray) -> tuple[float, float]:
    """SP = matrix diagonal; SEM over the N independent seeds."""
    diag = np.diag(mat)
    return float(diag.mean()), float(diag.std(ddof=1) / np.sqrt(len(diag)))


def xp_mean_sem(mat: np.ndarray) -> tuple[float, float]:
    """All-pairs XP mean and delete-one-seed jackknife SE (see xp_stats)."""
    theta, sem, _ = jackknife_xp(mat)
    return theta, sem


def main() -> None:
    labels: list[str] = []
    bases: list[str] = []
    sp_vals, sp_err, xp_vals, xp_err = [], [], [], []
    for label, fname, base in ROWS:
        path = MATRIX_DIR / fname
        if not path.exists():
            print(f"skip (no csv): {label.replace(chr(10), ' ')}  [{fname}]")
            continue
        mat = parse_xp_matrix(path)
        spm, spe = sp_mean_sem(mat)
        xpm, xpe = xp_mean_sem(mat)
        labels.append(label)
        bases.append(base)
        sp_vals.append(spm)
        sp_err.append(spe)
        xp_vals.append(xpm)
        xp_err.append(xpe)
        print(f"{label.replace(chr(10), ' '):20s} "
              f"SP {spm:.3f} +/- {spe:.4f}    XP {xpm:.3f} +/- {xpe:.4f}")

    x = np.arange(len(labels))
    w = 0.38
    ekw = dict(ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2)

    plt.rcParams["hatch.linewidth"] = 1.2
    fig, ax = plt.subplots(figsize=(2.6 + 1.8 * len(labels), 5.2))
    for i, base in enumerate(bases):
        ax.bar(x[i] - w / 2, sp_vals[i], w, color=base,
               edgecolor="none", yerr=sp_err[i], error_kw=ekw, zorder=3)
        ax.bar(x[i] + w / 2, xp_vals[i], w, color="white",
               hatch=XP_HATCH, edgecolor=base, linewidth=1.2,
               yerr=xp_err[i], error_kw=ekw, zorder=3)

    ax.axhline(RANDOM_BASELINE, ls="--", lw=1.8, color=RANDOM_COLOR, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Episode return", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor="#888888", edgecolor="none", label="Self-play (SP)"),
        Patch(facecolor="white", hatch=XP_HATCH, edgecolor="#888888",
              label="Cross-play (XP)"),
        Line2D([0], [0], color=RANDOM_COLOR, ls="--", lw=1.8,
               label="Random baseline"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="upper left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
