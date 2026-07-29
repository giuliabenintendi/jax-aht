"""Grouped SP/XP bar plot for the Overcooked v2 (demo_cook_simple) result.

Sized and styled to match the LBF OP figure (evaluation/lbf/plot_sp_xp_op.py): single
AAAI column width with a reduced (landscape) height, 9pt fonts at scale 1.0, and the
same bar / hatch / legend styling, so the two bar figures are uniform in the paper.

MATE = 1e-4, 12 seeds with s48 replaced by s123 (42-47, 49-53, 123).
SP = diagonal mean +/- std/sqrt(N);
XP = disjoint-pair mean +/- SE (Forkel et al. 2511.22581 eq 30/31), the paper
estimator, for every bar we trained (IPPO naked, OP, MATE, and the HE ENT 0.35
arm). The only exception is the Gessler et al. 2025 external reference (its
errors are the paper's std over 500 episodes).

Usage:
    uv run --no-project --with matplotlib --with numpy \\
        python -m evaluation.overcooked_v2.make_sp_xp_bar
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path("plots/overcooked_v2/ocv2_sp_xp_bar.png")

SP_HATCH = "////"
COL_W_IN = 3.3125  # AAAI column width
FIG_H_IN = 1.5     # compact landscape (aspect ~2.2), matching the LBF bar figure
YLABEL_PT = 9
XTICK_PT = 6.5
SUB_PT = 5       # small attribution line drawn beneath the primary x label
SUB_Y = -0.21    # axes-fraction y of the attribution line (below the primary tick label)
YTICK_PT = 8
LEGEND_PT = 9
VALUE_PT = 5

# (label, sub-label, base colour, SP mean, SP err, XP mean, XP err). Colours match the
# LBF / card-game palette (IPPO brown / OP blue / HE pink / MATE red) for cross-figure
# consistency. The Gessler et al. 2025 Other-Play bar is an external reference (yellow);
# its errors are the paper's std over 500 episodes, converted to SEM over 10 seeds
# (std / sqrt(10)). XP is disjoint-pair for our bars; IPPO and Gessler keep their
# original XP (see module docstring). HE = ENT 0.35, the worst-12 subset (by mean
# off-diagonal return) of its 16-seed pool, i.e. the conservative read of that arm.
COND = [
    ("IPPO", "", "#7D5A3C", 167.6, 8.3, 104.2, 17.8),
    ("OP", "(Gessler et al, 2025)", "#EAB308", 149.0, 25.0 / np.sqrt(10), 47.0, 48.0 / np.sqrt(10)),
    ("OP", "(Ours)", "#1f77b4", 131.8, 5.3, 116.8, 10.9),
    ("HE IPPO", "", "#D57DBF", 161.5, 8.3, 125.3, 14.4),
    ("MATE", "", "#d62728", 143.0, 1.3, 132.7, 2.5),
]


def render(cond: list[tuple], out: Path) -> None:
    labels = [c[0] for c in cond]
    subs = [c[1] for c in cond]
    bases = [c[2] for c in cond]
    sp_vals = [c[3] for c in cond]
    sp_err = [c[4] for c in cond]
    xp_vals = [c[5] for c in cond]
    xp_err = [c[6] for c in cond]

    x = np.arange(len(labels))
    w = 0.38
    ekw = dict(ecolor="black", elinewidth=0.5, capsize=1.8, capthick=0.5)

    plt.rcParams["hatch.linewidth"] = 0.7
    # Explicit margins (not constrained layout) so the reserved bottom band holds
    # both the primary tick label and the smaller attribution line at this short height.
    fig, ax = plt.subplots(figsize=(COL_W_IN, FIG_H_IN))
    fig.subplots_adjust(left=0.14, right=0.99, top=0.84, bottom=0.23)
    for i, base in enumerate(bases):
        ax.bar(x[i] - w / 2, sp_vals[i], w, color="white",
               hatch=SP_HATCH, edgecolor=base, linewidth=0.45,
               yerr=sp_err[i], error_kw=ekw, zorder=3)
        ax.bar(x[i] + w / 2, xp_vals[i], w, color=base,
               edgecolor="none", yerr=xp_err[i], error_kw=ekw, zorder=3)
        ax.text(x[i] - w / 2, sp_vals[i] + sp_err[i] + 2.5, f"{sp_vals[i]:.1f}",
                ha="center", va="bottom", fontsize=VALUE_PT, color="black")
        ax.text(x[i] + w / 2, xp_vals[i] + xp_err[i] + 2.5, f"{xp_vals[i]:.1f}",
                ha="center", va="bottom", fontsize=VALUE_PT, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=XTICK_PT)
    # Attribution drawn as its own small line just below the primary tick label, so
    # every primary label keeps the same size regardless of sub-label length.
    for i, sub in enumerate(subs):
        if sub:
            ax.text(x[i], SUB_Y, sub, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=SUB_PT, clip_on=False)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylabel("Episode return", fontsize=YLABEL_PT)
    ax.set_ylim(0, 235)
    ax.set_yticks(np.arange(0, 201, 50))
    ax.tick_params(axis="y", labelsize=YTICK_PT, width=0.5, length=2.2)
    ax.tick_params(axis="x", width=0.5, length=2.2)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor="white", hatch=SP_HATCH, edgecolor="#888888",
              label="Self-play (SP)"),
        Patch(facecolor="#888888", edgecolor="none", label="Cross-play (XP)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=LEGEND_PT, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.4, handletextpad=0.5, columnspacing=1.2, borderpad=0.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=1200)
    print(f"saved {out}")
    plt.close(fig)


def main() -> None:
    render(COND, OUT)


if __name__ == "__main__":
    main()
