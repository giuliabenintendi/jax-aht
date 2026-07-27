"""Combined SP/XP bar figure: card-agreement game and LBF side by side.

Reuses the per-env inlined summary numbers from
`evaluation.card_game.plot_sp_xp` (the IPPO / OP / MATE joint-attention story)
and `evaluation.lbf.plot_sp_xp_op`, so the bar values stay a single source of
truth. One shared legend on top; a sub-title below each panel
("Card Agreement Game", "LBF"). Each panel keeps its own y-axis because the
return scales differ (card return ceiling ~1, LBF ~0.5).

Usage:
    uv run --no-project --with matplotlib --with numpy \\
        python -m evaluation.plot_sp_xp_combined
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from evaluation.card_game.plot_sp_xp import (
    COND as CARD_COND,
    JA_LABELS,
    RANDOM_BASELINE,
    RANDOM_COLOR,
    XP_HATCH,
)
from evaluation.lbf.plot_sp_xp_op import COND as LBF_COND

OUT = Path("plots/sp_xp_bar_combined.png")

W = 0.38
EKW = dict(ecolor="black", elinewidth=1.4, capsize=5, capthick=1.4)


def draw_panel(ax, cond, ylim, yticks, baseline, title) -> None:
    labels = [c[0] for c in cond]
    x = np.arange(len(labels))
    for i, (_, base, spm, spe, xpm, xpe) in enumerate(cond):
        ax.bar(x[i] - W / 2, spm, W, color=base, edgecolor="none",
               yerr=spe, error_kw=EKW, zorder=3)
        ax.bar(x[i] + W / 2, xpm, W, color="white", hatch=XP_HATCH,
               edgecolor=base, linewidth=1.2, yerr=xpe, error_kw=EKW, zorder=3)
    if baseline is not None:
        ax.axhline(baseline, ls="--", lw=1.8, color=RANDOM_COLOR, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=24)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel("Episode return", fontsize=26)
    ax.set_xlabel(title, fontsize=26, labelpad=14)
    ax.tick_params(axis="y", labelsize=22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    card = [c for c in CARD_COND if c[0] in JA_LABELS]

    plt.rcParams["hatch.linewidth"] = 1.4
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.8))

    draw_panel(ax_l, card, (0, 1.12), np.arange(0, 1.01, 0.2),
               RANDOM_BASELINE, "Card Alignment Game")
    draw_panel(ax_r, LBF_COND, (0, 0.5), np.arange(0, 0.51, 0.1),
               None, "LBF")

    handles = [
        Patch(facecolor="#888888", edgecolor="none", label="Self-play (SP)"),
        Patch(facecolor="white", hatch=XP_HATCH, edgecolor="#888888",
              label="Cross-play (XP)"),
        Line2D([0], [0], color=RANDOM_COLOR, ls="--", lw=1.8,
               label="Random baseline"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=20, loc="upper center",
               bbox_to_anchor=(0.5, 1.03), ncol=3, handlelength=1.4,
               handletextpad=0.5, columnspacing=1.8, borderpad=0.0)

    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=3.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"saved {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
