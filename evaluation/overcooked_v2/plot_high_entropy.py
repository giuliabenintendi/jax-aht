"""High-entropy ZSC curve for OvercookedV2 (Forkel et al. entropy-coefficient figure).

Same house style as the LBF/card figures (evaluation/lbf/plot_high_entropy.py):
four curves vs entropy coefficient (SP/XP x greedy/nongreedy), shaded +/-SEM bands,
framed legend, linear/truthful x-axis. Values are episode return (not normalised), so
the y-axis spans 0..~200 rather than 0..0.4.

The sweep is 8 evenly-spaced coefficients (0.05..0.40, step 0.05). Only the alphas that
have data are drawn; the full grid stays as reserved ticks so the figure reads as a
template while the remaining arms finish. A single available alpha is drawn as a marker
with a SEM error bar (a shaded band needs two x-values); with two or more it switches to
the LBF line + fill_between look.

Reads per-alpha SP/XP (+SEM), greedy and sampled, from
results/overcooked_v2/high_entropy_curve.json.

Usage:
    uv run --no-project --with matplotlib --with numpy \\
        python evaluation/overcooked_v2/plot_high_entropy.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("results/overcooked_v2/high_entropy_curve.json")
OUT = Path("plots/overcooked_v2/high_entropy_curve_ocv2.png")
TITLE = "OvercookedV2"

# Planned sweep grid (8 values, step 0.05); reserved as ticks even before the arms land.
GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# (json key prefix, label, colour). SP = teal pair, XP = warm pair; greedy dark, nongreedy light.
CURVES = [
    ("sp_greedy", "SP greedy", "#0E6E7E"),     # dark teal
    ("sp_samp", "SP nongreedy", "#3FA9C2"),    # light teal
    ("xp_greedy", "XP greedy", "#E07C1F"),     # orange
    ("xp_samp", "XP nongreedy", "#C0341C"),    # brick red
]


def render(alphas: list[float], xs: np.ndarray, raw: dict, grid: list[float], out: Path,
           rotation: int = 0, legend_loc: str = "lower right") -> None:
    fig, ax = plt.subplots(figsize=(9, 2.6))
    single = len(alphas) == 1
    for key, label, color in CURVES:
        mean = np.array([raw[f"{a:g}"].get(key, np.nan) for a in alphas])
        sem = np.array([raw[f"{a:g}"].get(f"{key}_sem", 0.0) for a in alphas])
        if single:
            ax.errorbar(xs, mean, yerr=sem, fmt="o", color=color, label=label,
                        ms=7, capsize=4, elinewidth=1.6, capthick=1.6)
        else:
            ax.plot(xs, mean, "-", color=color, label=label, lw=2)
            ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)

    ax.set_ylim(bottom=0.0)
    ax.set_xlim(grid[0] - 0.02, grid[-1] + 0.02)
    ax.set_title(TITLE, fontsize=21)
    ax.set_xlabel("Entropy coefficient", fontsize=20)
    # 2x2 legend parked in the reserved (data-free) tail, flush with the last tick. A
    # bottom-left placement would cover both XP curves (the legend is over half the axes
    # height), and the tail is only 3.55in wide, so this one runs a point smaller and
    # tighter than the LBF/card legends to clear the 0.25 data end.
    leg = ax.legend(fontsize=14, loc=legend_loc, ncol=2, borderaxespad=0.0,
                    frameon=True, framealpha=0.6, edgecolor="0.85",
                    borderpad=0.25, labelspacing=0.25, columnspacing=0.6,
                    handletextpad=0.3, handlelength=1.0)
    leg.get_frame().set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", linewidth=0.6, color="#cccccc", alpha=0.8)
    ax.tick_params(labelsize=17)
    ax.set_yticks([0, 100, 200])
    ax.set_xticks(grid)
    ax.set_xticklabels([f"{a:g}" for a in grid], rotation=rotation,
                       ha=("right" if rotation else "center"), fontsize=17)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # Trim top/bottom whitespace but keep the 0.1" left/right pad.
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1, 0.0)
    fig.savefig(out, dpi=600, bbox_inches=bbox)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches=bbox)
    print(f"saved {out} (+ .pdf)")
    plt.close(fig)


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA} — run the sampled eval pass first to populate it")
    raw = json.loads(DATA.read_text())
    alphas = [a for a in GRID if raw.get(f"{a:g}")]
    render(alphas, np.array(alphas), raw, grid=GRID, out=OUT, rotation=0)


if __name__ == "__main__":
    main()
