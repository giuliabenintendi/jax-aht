"""High-entropy ZSC curve for the card game (Forkel et al. entropy-coefficient figure).

Four curves vs entropy coefficient, each with a shaded +/-SEM band:

- SP greedy   (blue)   — self-play, argmax decoding
- SP nongreedy(orange) — self-play, stochastic sampling
- XP greedy   (green)  — cross-play, argmax decoding
- XP nongreedy(red)    — cross-play, stochastic sampling

The signature result: as entropy rises, XP greedy climbs toward SP greedy while
the nongreedy curves peak then fall (sampling injects the trained-in stochasticity).

Reads per-alpha numbers from `results/card_game/high_entropy_curve.json`:

    {"0.5": {"sp_greedy": .., "sp_greedy_sem": .., "sp_samp": .., "sp_samp_sem": ..,
             "xp_greedy": .., "xp_greedy_sem": .., "xp_samp": .., "xp_samp_sem": ..}, ...}

SP/XP greedy come from training (`XP/sp_score`, `XP/xp_score_mean`); SP/XP nongreedy
from the post-hoc `run_xp_seeds.py --sampled` pass on the same best-checkpoint params.

Usage:
    uv run --no-project --with matplotlib --with numpy \\
        python -m evaluation.card_game.plot_high_entropy
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("results/card_game/high_entropy_curve.json")
OUT = Path("plots/card_game/high_entropy_curve_card_game.png")
RANDOM_BASELINE = 0.20
MIN_ALPHA = 0.25  # report from this coefficient onward
TITLE = "Card Alignment Game"

# (json key prefix, label, colour). SP = teal pair, XP = red pair; greedy dark, nongreedy light.
CURVES = [
    ("sp_greedy", "SP greedy", "#0E6E7E"),     # dark teal
    ("sp_samp", "SP nongreedy", "#3FA9C2"),    # light teal
    ("xp_greedy", "XP greedy", "#E07C1F"),     # orange
    ("xp_samp", "XP nongreedy", "#C0341C"),    # brick red
]


def render(alphas: list[float], xs: np.ndarray, raw: dict, logx: bool, out: Path, rotation: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for key, label, color in CURVES:
        mean = np.array([raw[f"{a:g}"][key] for a in alphas])
        sem = np.array([raw[f"{a:g}"].get(f"{key}_sem", 0.0) for a in alphas])
        ax.plot(xs, mean, "-", color=color, label=label, lw=2)
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)

    ax.axhline(RANDOM_BASELINE, ls="--", lw=1.4, color="#888888", zorder=1)
    if logx:
        ax.set_xscale("log")
    ax.set_title(TITLE, fontsize=21)
    leg = ax.legend(fontsize=15, loc="center right", frameon=True, framealpha=0.6,
                    edgecolor="0.85", borderpad=0.3, labelspacing=0.25,
                    handletextpad=0.4, handlelength=1.4)
    leg.get_frame().set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", linewidth=0.6, color="#cccccc", alpha=0.8)
    ax.tick_params(labelsize=17)
    # Squat panel: three y-ticks, one of them on the chance line so it reads off the axis.
    ax.set_yticks([RANDOM_BASELINE, 0.6, 1.0])
    # Tick only the entropy coefficients we actually tested (no auto ticks).
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a:g}" for a in alphas], rotation=rotation,
                       ha=("right" if rotation else "center"), fontsize=17)
    if logx:
        ax.minorticks_off()

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
    # Even 0.25-step grid; every point is a real run at its stated coefficient.
    even = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    alphas = [a for a in even if raw.get(f"{a:g}")]
    render(alphas, np.array(alphas), raw, logx=False, out=OUT, rotation=0)


if __name__ == "__main__":
    main()
