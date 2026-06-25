"""Grouped SP/XP bar plot for the LBF generalization result.

Two conditions, matched 3M / 8-seed budget on the 50-step task:

- Baseline: naked IPPO (no partner-attention feed, no aux) — run 32xt8ojt.
- JA: dense attended-object occupancy (feed + aux) — run ycghndhm.

Bar values are computed from the cross-play matrices in `xp_results/` via
`evaluation.card_game.xp_stats`: SP = diagonal mean with SEM over seeds, XP =
all-pairs off-diagonal mean with delete-one-seed SE (same estimator as the
card-game figure). The matrices live on the GPU box; the numbers are inlined
here so the figure renders without them.

One base colour per condition; SP = solid fill, XP = white column with a
diagonal hatch in the same colour. No random-baseline line.

Usage:
    uv run --no-project --with matplotlib --with numpy \\
        python -m evaluation.lbf.plot_sp_xp
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path("plots/lbf/sp_xp_bar.png")

XP_HATCH = "//"

# (label, base colour, SP mean, SP sem, XP mean, XP sem). Computed from the 3M
# 8-seed cross-play matrices (xp_stats.py): naked baseline 32xt8ojt, dense
# attended-object JA ycghndhm. Green baseline / red JA.
COND = [
    ("Baseline", "#56C490", 0.404, 0.029, 0.239, 0.048),  # naked IPPO (32xt8ojt)
    ("JA",       "#FF5669", 0.483, 0.005, 0.417, 0.023),  # dense-object feed+aux (ycghndhm)
]


def render(cond: list[tuple], out: Path) -> None:
    labels = [c[0] for c in cond]
    bases = [c[1] for c in cond]
    sp_vals = [c[2] for c in cond]
    sp_err = [c[3] for c in cond]
    xp_vals = [c[4] for c in cond]
    xp_err = [c[5] for c in cond]
    for label, _, spm, spe, xpm, xpe in cond:
        print(f"{label:10s} SP {spm:.3f}+/-{spe:.3f}   XP {xpm:.3f}+/-{xpe:.3f}")

    x = np.arange(len(labels))
    w = 0.38
    ekw = dict(ecolor="black", elinewidth=1.4, capsize=5, capthick=1.4)

    plt.rcParams["hatch.linewidth"] = 1.4
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for i, base in enumerate(bases):
        ax.bar(x[i] - w / 2, sp_vals[i], w, color=base,
               edgecolor="none", yerr=sp_err[i], error_kw=ekw, zorder=3)
        ax.bar(x[i] + w / 2, xp_vals[i], w, color="white",
               hatch=XP_HATCH, edgecolor=base, linewidth=1.2,
               yerr=xp_err[i], error_kw=ekw, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylabel("Episode return", fontsize=20)
    ax.set_ylim(0, 0.5)  # LBF return ceiling (force_coop, normalized)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # JA's SP bar nearly reaches the 0.5 ceiling, so place the legend above the axes.
    handles = [
        Patch(facecolor="#888888", edgecolor="none", label="Self-play (SP)"),
        Patch(facecolor="white", hatch=XP_HATCH, edgecolor="#888888",
              label="Cross-play (XP)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=13, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.4, handletextpad=0.5, columnspacing=1.4, borderpad=0.0)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def main() -> None:
    render(COND, OUT)


if __name__ == "__main__":
    main()
