"""SP/XP bar figures for all three envs as matched LaTeX subfigures.

Emits one file per env (card game, LBF, ocv2) with identical height and vertical
margins so the axes align when placed side by side in a subfigure row; widths sum
to the AAAI textwidth (3.47 + 1.68 + 1.84 = 6.99 in), derived from the measured
5.5 pt widths of each figure's widest adjacent tick-label pair so every
single-line label fits. The legend is drawn only on the leftmost (card) figure.

LBF and ocv2 numbers are imported from their source scripts
(evaluation.lbf.plot_sp_xp_op, plots/overcooked_v2/make_sp_xp_bar.py); the card
numbers are copied here because plot_ablation_bars.py plots at import time and
cannot be imported without side effects -- keep the two lists in sync.

Tick labels are horizontal and single-line; the ocv2 OP attribution is a smaller
separate line under the tick label. Panel naming is left to the LaTeX
subcaptions.

LBF is the panel where the value labels crowd, so it gets the width the y-axis
text of all three panels gives back at 8 pt / 6 pt plus its own trimmed x
margins: 0.351 in of group pitch against the 0.289 in it had at 1.58 in wide.

Usage:
    uv run --no-project --with matplotlib --with numpy \\
        python -m evaluation.plot_sp_xp_all
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from evaluation.lbf.plot_sp_xp_op import COND as LBF_COND
from plots.overcooked_v2.make_sp_xp_bar import COND as OCV2_COND

OUT_DIR = Path("plots")

SP_HATCH = "////"
RANDOM_COLOR, RANDOM_BASELINE = "#8B0000", 0.20

FIG_H_IN = 1.55
# Same top/bottom fractions everywhere so the axes align across the subfigure row.
TOP_FRAC, BOTTOM_FRAC = 0.88, 0.22
YLABEL_PT = 8
YTICK_PT = 6
# Largest size at which every label stays on a single line: the figure widths
# below put the group pitch just above each figure's widest adjacent label pair
# measured at 5.5 pt (pitch = axes_width / (n_groups + 0.4) for the xlim margins).
XTICK_PT = 5.5
SUB_PT = 3.5         # ocv2 attribution line under the two OP tick labels
ABL_TICK_PT = 5      # "w/o X" labels; smaller so the three fit side by side
BRACE_PT = 5.5       # "MATE ablations" bracket label
LEGEND_PT = 7.5
VALUE_PT = 4.5

W = 0.38
EKW = dict(ecolor="black", elinewidth=0.5, capsize=1.8, capthick=0.5)

# Copied from evaluation/card_game/plot_ablation_bars.py (see module docstring).
# (label, colour, SP, SPsem, XP, XPsem) -- HE = highest all-pairs XP over the
# 48-seed entropy sweep rerun of 2026-07-27 (alpha=2; archived cardGame/HE/2).
CARD_COND = [
    ("IPPO",       "#7D5A3C", 1.000, 0.001, 0.167, 0.078),
    ("OP",         "#1f77b4", 0.200, 0.002, 0.201, 0.002),
    ("HE IPPO",    "#e377c2", 1.000, 0.000, 0.209, 0.006),
    ("Lee et al.", "#8EBB69", 1.000, 0.000, 0.195, 0.002),
    ("w/o OP",     "#ff7f0e", 1.000, 0.000, 0.218, 0.028),
    ("w/o feed",   "#9467bd", 0.203, 0.058, 0.200, 0.004),
    ("w/o aux",    "#3E8347", 0.219, 0.060, 0.203, 0.005),
    ("MATE",       "#d62728", 0.872, 0.008, 0.823, 0.010),
]
CARD_ABLATION_IDX = (4, 5, 6)


def draw_underbrace(ax, xmin, xmax, y_top, depth, transform, beta=20.0, lw=0.5):
    """Horizontal curly underbrace (central tip pointing down) grouping xmin..xmax."""
    n = 201
    x = np.linspace(xmin, xmax, n)
    xh = x[: n // 2 + 1]
    yh = 1 / (1 + np.exp(-beta * (xh - xh[0]))) + 1 / (1 + np.exp(-beta * (xh - xh[-1])))
    yfull = np.concatenate((yh, yh[-2::-1]))  # 0.5 at ends, 1.5 at centre
    y = y_top - depth * (yfull - 0.5)
    ax.plot(x, y, color="black", lw=lw, transform=transform, clip_on=False)


def new_fig(width_in: float, left_in: float):
    """Figure/axes with the shared vertical layout; horizontal margins in inches."""
    fig, ax = plt.subplots(figsize=(width_in, FIG_H_IN))
    fig.subplots_adjust(left=left_in / width_in, right=1 - 0.03 / width_in,
                        top=TOP_FRAC, bottom=BOTTOM_FRAC)
    return fig, ax


def draw_panel(ax, cond, *, ylim, yticks, value_fmt, value_pad,
               baseline=None, ylabel=False, xmargin=0.7):
    """One SP/XP group per condition; `cond` = (label, colour, sp, spe, xp, xpe)."""
    x = np.arange(len(cond))
    for i, (_, base, spm, spe, xpm, xpe) in enumerate(cond):
        ax.bar(x[i] - W / 2, spm, W, color="white", hatch=SP_HATCH,
               edgecolor=base, linewidth=0.45, yerr=spe, error_kw=EKW, zorder=3)
        ax.bar(x[i] + W / 2, xpm, W, color=base, edgecolor="none",
               yerr=xpe, error_kw=EKW, zorder=3)
        # Nudged outward so near-equal SP/XP values cannot collide.
        ax.text(x[i] - W / 2 - 0.07, spm + spe + value_pad, value_fmt % spm,
                ha="center", va="bottom", fontsize=VALUE_PT, color="black")
        ax.text(x[i] + W / 2 + 0.07, xpm + xpe + value_pad, value_fmt % xpm,
                ha="center", va="bottom", fontsize=VALUE_PT, color="black")
    if baseline is not None:
        ax.axhline(baseline, ls="--", lw=0.7, color=RANDOM_COLOR, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cond], fontsize=XTICK_PT)
    ax.set_xlim(-xmargin, len(cond) - 1 + xmargin)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    if ylabel:
        ax.set_ylabel("Episode return", fontsize=YLABEL_PT)
    ax.tick_params(axis="y", labelsize=YTICK_PT, width=0.5, length=2.2)
    ax.tick_params(axis="x", width=0.5, length=2.2)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name: str) -> None:
    out = OUT_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=1200)
    print(f"saved {out}")
    plt.close(fig)


def main() -> None:
    plt.rcParams["hatch.linewidth"] = 0.7

    # Card game: legend, random-baseline line, ablation bracket.
    fig, ax = new_fig(3.47, 0.39)
    draw_panel(ax, CARD_COND, ylim=(0, 1.12), yticks=np.arange(0, 1.01, 0.2),
               value_fmt="%.2f", value_pad=0.02,
               baseline=RANDOM_BASELINE, ylabel=True)
    for i, t in enumerate(ax.get_xticklabels()):
        if i in CARD_ABLATION_IDX:
            t.set_fontsize(ABL_TICK_PT)
    xtrans = ax.get_xaxis_transform()
    lo, hi = CARD_ABLATION_IDX[0] - 0.3, CARD_ABLATION_IDX[-1] + 0.3
    draw_underbrace(ax, lo, hi, -0.15, 0.05, xtrans)
    ax.text((lo + hi) / 2, -0.24, "MATE ablations", transform=xtrans,
            ha="center", va="top", fontsize=BRACE_PT)
    handles = [
        Patch(facecolor="white", hatch=SP_HATCH, edgecolor="#888888",
              label="Self-play (SP)"),
        Patch(facecolor="#888888", edgecolor="none", label="Cross-play (XP)"),
        Line2D([0], [0], color=RANDOM_COLOR, ls="--", lw=0.7,
               label="Random baseline"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=LEGEND_PT,
               loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, handlelength=0.9, handletextpad=0.3, columnspacing=0.6,
               borderpad=0.0)
    save(fig, "sp_xp_subfig_card.png")

    # LBF: no legend, no baseline (random return ~0). Widest of the three panels
    # relative to its four groups -- it takes the width freed by the smaller y-axis
    # text, and its x margins are trimmed to what the end tick labels need, so the
    # near-equal MATE values keep a clear gap side by side.
    fig, ax = new_fig(1.68, 0.21)
    draw_panel(ax, LBF_COND, ylim=(0, 0.55), yticks=np.arange(0, 0.51, 0.1),
               value_fmt="%.2f", value_pad=0.012, xmargin=0.55)
    save(fig, "sp_xp_subfig_lbf.png")

    # Ocv2: attribution as a smaller line under the two OP tick labels.
    # Values as integers: the panel has no room for a decimal digit.
    ocv2_sub = {"(Gessler et al, 2025)": "(Gessler et al.)", "(Ours)": "(Ours)"}
    fig, ax = new_fig(1.84, 0.24)
    draw_panel(ax, [(c[0], *c[2:]) for c in OCV2_COND],
               ylim=(0, 205), yticks=np.arange(0, 201, 50),
               value_fmt="%.0f", value_pad=5.0)
    for i, sub in enumerate(c[1] for c in OCV2_COND):
        if sub:
            ax.text(i, -0.16, ocv2_sub[sub], transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=SUB_PT, clip_on=False)
    save(fig, "sp_xp_subfig_ocv2.png")


if __name__ == "__main__":
    main()
