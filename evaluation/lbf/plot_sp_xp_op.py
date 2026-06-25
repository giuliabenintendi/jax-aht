"""Grouped SP/XP bar plot for the LBF Other-Play / MATE result.

Three conditions, matched 3M / 8-seed budget on the 50-step task, mirroring the
card-game OP figure:

- Self Play: naked IPPO, no Other-Play, no aux — run 32xt8ojt (lbf50_baseline_3M_8s).
- Other Play: Other-Play mirror, no aux (naked under OP) — run lbf50_opmirror_8s_3M.
- MATE: Other-Play + attended-object occupancy aux (feed + aux) — run lbf50_opja_8s_3M.

Bar values are computed from the cross-play matrices in `xp_results/` via
`evaluation.card_game.xp_stats`: SP = diagonal mean with SEM over seeds, XP =
all-pairs off-diagonal mean with delete-one-seed SE. The matrices live on the
GPU box; the numbers are inlined here so the figure renders without them.

One base colour per condition; SP = solid fill, XP = white column with a
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

XP_HATCH = "//"

# (label, base colour, SP mean, SP sem, XP mean, XP sem). Computed from the 3M
# 8-seed cross-play matrices (xp_stats.py). IPPO lavender / OP blue / MATE orange
# — matching the card-game OP figure palette.
COND = [
    ("IPPO", "#B7B6E1", 0.404, 0.027, 0.239, 0.048),  # naked IPPO, no OP (32xt8ojt)
    ("OP",   "#457BE8", 0.296, 0.053, 0.255, 0.073),  # OP only (lbf50_opmirror_8s_3M)
    ("MATE", "#E68D3C", 0.461, 0.003, 0.439, 0.007),  # OP + JA (lbf50_opja_8s_3M)
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
    ekw = dict(ecolor="black", elinewidth=1.4, capsize=5, capthick=1.4)

    plt.rcParams["hatch.linewidth"] = 1.4
    fig, ax = plt.subplots(figsize=(2.6 + 1.6 * len(labels), 5.4))
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
