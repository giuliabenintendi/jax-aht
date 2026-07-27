"""High-entropy ZSC curve for LBF (Forkel et al. entropy-coefficient figure).

Same house style as the card-game figure (evaluation/card_game/plot_high_entropy.py):
four curves vs entropy coefficient (SP/XP x greedy/nongreedy), no markers, shaded
+/-SEM bands, framed center-right legend, linear/truthful x-axis. Unlike the card game,
LBF has no clean chance baseline, so the dashed random-baseline line is omitted; every
curve peaks (~0.15) then collapses.

Reads per-alpha SP/XP (+SEM), greedy and sampled, from
results/lbf/lbf_high_entropy_curve.json (built by the LBF collector).

Usage:
    uv run --no-project --with matplotlib --with numpy \\
        python evaluation/lbf/plot_high_entropy.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

DATA = Path("results/lbf/lbf_high_entropy_curve.json")
OUT = Path("plots/lbf/high_entropy_curve_LBF.png")
TITLE = "LBF"

# (json key prefix, label, colour). SP = teal pair, XP = warm pair; greedy dark, nongreedy light.
CURVES = [
    ("sp_greedy", "SP greedy", "#0E6E7E"),     # dark teal
    ("sp_samp", "SP nongreedy", "#3FA9C2"),    # light teal
    ("xp_greedy", "XP greedy", "#E07C1F"),     # orange
    ("xp_samp", "XP nongreedy", "#C0341C"),    # brick red
]


def render(alphas: list[float], xs: np.ndarray, raw: dict, out: Path, rotation: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for key, label, color in CURVES:
        mean = np.array([raw[f"{a:g}"].get(key, np.nan) for a in alphas])
        sem = np.array([raw[f"{a:g}"].get(f"{key}_sem", 0.0) for a in alphas])
        ax.plot(xs, mean, "-", color=color, label=label, lw=2)
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)

    ax.set_ylim(bottom=0.0)
    ax.set_title(TITLE, fontsize=21)
    # 2x2 legend in the empty lower-left corner (no curve drops below 0.25 there),
    # its left edge flush with the first coefficient tick.
    leg_tr = blended_transform_factory(ax.transData, ax.transAxes)
    leg = ax.legend(fontsize=15, loc="lower left", ncol=2,
                    bbox_to_anchor=(xs[0], 0.04), bbox_transform=leg_tr, borderaxespad=0.0,
                    frameon=True, framealpha=0.6, edgecolor="0.85",
                    borderpad=0.3, labelspacing=0.25, columnspacing=1.0,
                    handletextpad=0.4, handlelength=1.4)
    leg.get_frame().set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", linewidth=0.6, color="#cccccc", alpha=0.8)
    ax.tick_params(labelsize=17)
    ax.set_yticks([0.0, 0.2, 0.4])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a:g}" for a in alphas], rotation=rotation,
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
        raise SystemExit(f"missing {DATA} — run the LBF collector first to populate it")
    raw = json.loads(DATA.read_text())
    # Even-grid peak-zoom (0.075..0.25, step 0.025); plot the points we actually have
    # at their true linear positions, so the 0.2/0.225 gap shows honestly.
    even = [0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    alphas = [a for a in even if raw.get(f"{a:g}")]
    render(alphas, np.array(alphas), raw, out=OUT, rotation=0)


if __name__ == "__main__":
    main()
