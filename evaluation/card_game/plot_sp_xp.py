"""Grouped SP/XP bar plots for the card-game results.

Renders two presentation figures from per-condition summary numbers:

- `sp_xp_bar_all.png`: all six conditions.
- `sp_xp_bar_ja.png`: the joint-attention story only
  (SP, OP only, OP + JA, OP + JA + shaping).

Bar values:

- OP-condition SP/XP are computed from the cross-play matrices in
  `xp_matrices/` via `xp_stats.py`: SP = diagonal mean (SEM over seeds),
  XP = all-pairs off-diagonal mean with delete-one-seed SE. The matrices
  live on the GPU box; the numbers are inlined here so the figure renders
  without them.
- The self-play control (no OP): self-play solves the game (SP = 1.0); its
  cross-play sits near chance because two independently-trained policies
  share a convention only ~1/5 of the time (5 cards). Measured over the
  48-seed no-OP run.

One base colour per condition; SP = solid fill, XP = white column with a
diagonal hatch in the same colour. Dashed random-baseline line, legend
centred on top.

Usage:
    uv run --no-project --with matplotlib --with numpy \\
        python -m evaluation.card_game.plot_sp_xp
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT_ALL = Path("plots/card_game/sp_xp_bar_all.png")
OUT_JA = Path("plots/card_game/sp_xp_bar_ja.png")

RANDOM_COLOR = "#d62728"
RANDOM_BASELINE = 0.20
XP_HATCH = "//"

# (label, base colour, SP mean, SP sem, XP mean, XP sem).
# OP rows are computed from xp_matrices/ via xp_stats.py (SP = diagonal mean,
# SEM over seeds; XP = all-pairs off-diagonal mean, delete-one-seed SE). The
# self-play control is the measured 48-seed no-OP run.
# Unshaped block (SP, OP only, OP + JA, OP + comm) then shaped block.
COND = [
    ("SP",                   "#B7B6E5", 1.000, 0.001, 0.211, 0.059),  # lilac control (no OP); 48-seed noop_1M_48s
    ("OP only",              "#2E7DF0", 0.200, 0.002, 0.200, 0.002),  # blue (chance)
    ("OP + JA",              "#F5871F", 0.872, 0.008, 0.832, 0.008),  # orange; qvublwxp best-ckpt (48 seeds)
    ("OP + comm",            "#8C82F6", 0.344, 0.027, 0.355, 0.028),  # purple (no shaping)
    ("OP + JA\n+ shaping",   "#51B18D", 0.984, 0.002, 0.933, 0.007),  # green; hm3x0pdv best-ckpt (48 seeds)
    ("OP + comm\n+ shaping", "#F55F74", 0.847, 0.038, 0.874, 0.032),  # pink
]

# Joint-attention story: SP control, OP baseline, OP + JA (no shaping bar).
JA_LABELS = {"SP", "OP only", "OP + JA"}


def render(cond: list[tuple], out: Path) -> None:
    labels = [c[0] for c in cond]
    bases = [c[1] for c in cond]
    sp_vals = [c[2] for c in cond]
    sp_err = [c[3] for c in cond]
    xp_vals = [c[4] for c in cond]
    xp_err = [c[5] for c in cond]
    for label, _, spm, spe, xpm, xpe in cond:
        print(f"{label.replace(chr(10), ' '):24s} "
              f"SP {spm:.3f}+/-{spe:.3f}   XP {xpm:.3f}+/-{xpe:.3f}")

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
    # Centre the legend in the open gap between the SP condition's bars and the
    # next tall bar, i.e. over the low OP-only column (x = op_idx), leaving a
    # small margin on each side. The OP-only bars are low, so nothing is hidden.
    op_idx = labels.index("OP only") if "OP only" in labels else 1
    ax.legend(handles=handles, frameon=False, fontsize=13, loc="upper center",
              bbox_to_anchor=(op_idx - 0.15, 1.08), bbox_transform=ax.transData,
              handlelength=1.4, handletextpad=0.5, labelspacing=0.3, borderpad=0.0)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}\n")
    plt.close(fig)


def main() -> None:
    render(COND, OUT_ALL)
    render([c for c in COND if c[0] in JA_LABELS], OUT_JA)


if __name__ == "__main__":
    main()
