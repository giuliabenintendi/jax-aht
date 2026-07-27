"""Grouped SP/XP bar plot for the LBF Other-Play / MATE result.

Four conditions, matched 3M / 12-seed budget on the 50-step task, mirroring the
card-game OP figure:

- IPPO: naked IPPO, no Other-Play, no aux — run xcwlk81p (lbf50_baseline_12s_3M).
- OP: Other-Play mirror, no aux (naked under OP) — run koftp3iw (lbf50_oponly_12s_3M).
- HE IPPO: best high-entropy IPPO (ENT_COEF=0.15, no OP/aux) — run lbf_he_a015_3M_12s.
- MATE: Other-Play + attended-object occupancy aux (feed + aux) — run 0cjgqcj6 (lbf50_opja_12s_3M).

Bar values are computed from the cross-play matrices in `xp_results/` via
`evaluation.card_game.xp_stats`: SP = diagonal mean with SEM over seeds, XP =
disjoint-pair mean + SE (`xp_mean_se`, Forkel 2511.22581 eq 30/31). The matrices
live on the GPU box; the numbers are inlined here so the figure renders without
them.

One base colour per condition; XP = solid fill, SP = white column with a
diagonal hatch in the same colour. No random-baseline line (LBF random return
is ~0, so the reference is uninformative).

Usage:
    uv run --no-project --with matplotlib --with numpy \\
        python -m evaluation.lbf.plot_sp_xp_op
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path("plots/lbf/sp_xp_bar_op.png")

SP_HATCH = "////"

# Single-column figure (\includegraphics[width=\linewidth], \linewidth = column width),
# so the point sizes below render at scale 1.0. With four groups the bars are narrower
# than the three-group ocv2 figure, so the height is reduced to keep a landscape aspect.
COL_W_IN = 3.3125  # AAAI column width = (textwidth 7.0 - columnsep 0.375) / 2
FIG_H_IN = 1.5     # compact landscape (aspect ~2.2) so the 4-group panel reads small
YLABEL_PT = 9
XTICK_PT = 9
YTICK_PT = 8
LEGEND_PT = 9
VALUE_PT = 5  # value numbers on top of each bar, matching the ocv2 figure

# (label, base colour, SP mean, SP sem, XP mean, XP sem). Computed from the 3M
# 12-seed cross-play matrices (xp_stats.py: SP = diagonal mean, SEM over seeds;
# XP = disjoint-pair mean + SE, xp_stats.xp_mean_se, Forkel 2511.22581 eq 30/31).
# IPPO brown / OP blue / MATE red — matching the card-game ablation palette.
COND = [
    ("IPPO", "#7D5A3C", 0.356, 0.057, 0.217, 0.072),  # IPPO objective on shared JA net, no OP/aux (xcwlk81p)
    ("OP",   "#1f77b4", 0.263, 0.062, 0.221, 0.087),  # OP only (koftp3iw)
    ("HE IPPO", "#e377c2", 0.400, 0.036, 0.328, 0.057),  # best HE, ENT_COEF=0.15, no OP/aux (lbf_he_a015_3M_12s)
    ("MATE", "#d62728", 0.460, 0.002, 0.446, 0.005),  # OP + JA feed+aux (0cjgqcj6)
]


def render(cond: list[tuple], out: Path) -> None:
    labels = [c[0] for c in cond]
    bases = [c[1] for c in cond]
    sp_vals = [c[2] for c in cond]
    sp_err = [c[3] for c in cond]
    xp_vals = [c[4] for c in cond]
    xp_err = [c[5] for c in cond]
    for label, _, spm, spe, xpm, xpe in cond:
        print(f"{label:11s} SP {spm:.3f}+/-{spe:.3f}   XP {xpm:.3f}+/-{xpe:.3f}")

    x = np.arange(len(labels))
    w = 0.38
    ekw = dict(ecolor="black", elinewidth=0.5, capsize=1.8, capthick=0.5)

    plt.rcParams["hatch.linewidth"] = 0.7  # keep the original hatch weight (liked)
    fig, ax = plt.subplots(figsize=(COL_W_IN, FIG_H_IN), layout="constrained")
    for i, base in enumerate(bases):
        ax.bar(x[i] - w / 2, sp_vals[i], w, color="white",
               hatch=SP_HATCH, edgecolor=base, linewidth=0.45,
               yerr=sp_err[i], error_kw=ekw, zorder=3)
        ax.bar(x[i] + w / 2, xp_vals[i], w, color=base,
               edgecolor="none", yerr=xp_err[i], error_kw=ekw, zorder=3)
        ax.text(x[i] - w / 2, sp_vals[i] + sp_err[i] + 0.01, f"{sp_vals[i]:.2f}",
                ha="center", va="bottom", fontsize=VALUE_PT, color="black")
        ax.text(x[i] + w / 2, xp_vals[i] + xp_err[i] + 0.01, f"{xp_vals[i]:.2f}",
                ha="center", va="bottom", fontsize=VALUE_PT, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=XTICK_PT)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylabel("Episode return", fontsize=YLABEL_PT)
    ax.set_ylim(0, 0.55)  # 0.5 return ceiling + headroom for the value labels
    ax.set_yticks(np.arange(0, 0.51, 0.1))
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
    fig.savefig(out, dpi=1200)  # high-res PNG for direct \includegraphics
    print(f"saved {out}")
    plt.close(fig)


def main() -> None:
    render(COND, OUT)


if __name__ == "__main__":
    main()
