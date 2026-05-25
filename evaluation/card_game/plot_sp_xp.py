"""Grouped SP/XP bar plot for the card-game results table.

One base colour per condition; SP = solid fill, XP = white column with a
diagonal hatch in the same colour. Dashed random-baseline line on top,
legend top-left. Significance stars above each XP bar (paired t-test on
m = N/2 disjoint pairs vs OP-only).

Bars and error bars are computed from each run's cross-play matrix
(`xp_score_matrix.csv` in `xp_matrices/`):
  SP height = mean of the matrix diagonal (seed i paired with itself)
  SP error  = std(diagonal, ddof=1) / sqrt(N)        -- N independent seeds
  XP height = mean of every off-diagonal cell (all cross-play pairs)
  XP error  = delete-one-seed SE of that all-pairs mean (see xp_stats.py)
  XP star   = paired t-test (one-sided, greater) on disjoint-pair samples
              vs the OP-only matrix:  *** p<0.001, ** p<0.01, * p<0.05

Usage:
    uv run --no-project --with matplotlib --with numpy --with scipy \\
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

from evaluation.card_game.xp_stats import (
    parse_xp_matrix,
    paired_test_vs_baseline,
    stars,
    xp_mean_se,
)

MATRIX_DIR = Path(__file__).parent / "xp_matrices"
OUT = Path("plots/card_game/sp_xp_bar.png")

RANDOM_COLOR = "#d62728"
RANDOM_BASELINE = 0.20
XP_HATCH = "//"

# (display label, csv filename, base colour)
ROWS = [
    ("OP only",                "op_only.csv",          "#2E7DF0"),  # blue
    ("OP + JA",                "op_ja.csv",            "#F5871F"),  # orange
    ("OP + comm",              "op_comm_noshape.csv",  "#8C82F6"),  # lilla
    ("OP + JA\n+ shaping",     "op_ja_shaping.csv",    "#51B18D"),  # green
    ("OP + comm\n+ match",     "op_comm_match.csv",    "#F77F92"),  # medium pink
    ("OP + comm\n+ shaping",   "op_comm_shaping.csv",  "#F55F74"),  # Abet 823 HR-LAQ
]
BASELINE_LABEL = "OP only"


def sp_mean_sem(mat: np.ndarray) -> tuple[float, float]:
    """SP = matrix diagonal; SEM over the N independent seeds."""
    diag = np.diag(mat)
    return float(diag.mean()), float(diag.std(ddof=1) / np.sqrt(len(diag)))


def xp_mean_sem(mat: np.ndarray) -> tuple[float, float]:
    """All-pairs XP mean and delete-one-seed SE (see xp_stats)."""
    theta, sem, _ = xp_mean_se(mat)
    return theta, sem


def main() -> None:
    # Load all available matrices first
    mats: dict[str, np.ndarray] = {}
    plot_rows: list[tuple[str, str, str]] = []
    for label, fname, base in ROWS:
        path = MATRIX_DIR / fname
        if not path.exists():
            print(f"skip (no csv): {label.replace(chr(10), ' ')}  [{fname}]")
            continue
        mats[label] = parse_xp_matrix(path)
        plot_rows.append((label, fname, base))

    baseline_mat = mats.get(BASELINE_LABEL)
    if baseline_mat is None:
        print(f"WARNING: baseline {BASELINE_LABEL!r} missing — no significance stars.")

    labels: list[str] = []
    bases: list[str] = []
    sp_vals, sp_err, xp_vals, xp_err, sig_marks = [], [], [], [], []
    for label, fname, base in plot_rows:
        mat = mats[label]
        spm, spe = sp_mean_sem(mat)
        xpm, xpe = xp_mean_sem(mat)

        mark = ""
        if baseline_mat is not None and label != BASELINE_LABEL:
            if mat.shape == baseline_mat.shape:
                r = paired_test_vs_baseline(mat, baseline_mat, alternative="greater")
                mark = stars(r["p"])
                print(f"{label.replace(chr(10), ' '):24s} "
                      f"SP {spm:.3f}+/-{spe:.4f}   XP {xpm:.3f}+/-{xpe:.4f}   "
                      f"Δ={r['mean_diff']:+.3f}  t={r['t']:.2f}  p={r['p']:.4g}  {mark}")
            else:
                print(f"{label.replace(chr(10), ' '):24s} "
                      f"SP {spm:.3f}+/-{spe:.4f}   XP {xpm:.3f}+/-{xpe:.4f}   "
                      f"(shape {mat.shape} != baseline {baseline_mat.shape}, no test)")
        else:
            print(f"{label.replace(chr(10), ' '):24s} "
                  f"SP {spm:.3f}+/-{spe:.4f}   XP {xpm:.3f}+/-{xpe:.4f}   (baseline)")

        labels.append(label)
        bases.append(base)
        sp_vals.append(spm); sp_err.append(spe)
        xp_vals.append(xpm); xp_err.append(xpe)
        sig_marks.append(mark)

    x = np.arange(len(labels))
    w = 0.38
    ekw = dict(ecolor="black", elinewidth=1.4, capsize=5, capthick=1.4)

    plt.rcParams["hatch.linewidth"] = 1.4
    fig, ax = plt.subplots(figsize=(2.6 + 1.6 * len(labels), 5.4))
    for i, base in enumerate(bases):
        ax.bar(x[i] - w / 2, sp_vals[i], w, color=base,
               edgecolor="none", yerr=sp_err[i], error_kw=ekw, zorder=3)
        ax.bar(x[i] + w / 2, xp_vals[i], w, color="white",
               hatch=XP_HATCH, edgecolor=base, linewidth=1.2,
               yerr=xp_err[i], error_kw=ekw, zorder=3)
    ax.axhline(RANDOM_BASELINE, ls="--", lw=1.8, color=RANDOM_COLOR, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_ylabel("Episode return", fontsize=20)
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor="#888888", edgecolor="none", label="Self-play (SP)"),
        Patch(facecolor="white", hatch=XP_HATCH, edgecolor="#888888",
              label="Cross-play (XP)"),
        Line2D([0], [0], color=RANDOM_COLOR, ls="--", lw=1.8,
               label="Random baseline"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=16, loc="upper left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
