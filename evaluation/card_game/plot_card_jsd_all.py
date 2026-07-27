"""All-condition card-level JSD-over-episode curves (companion to the
ablation bars). One figure overlaying every condition in the ablation-bar
palette, in the same house style as `plot_card_jsd.py`.

Reads `<data-dir>/<key>.npz` produced by `eval_card_jsd.py` for each condition.

Usage:
    python -m evaluation.card_game.plot_card_jsd_all \\
        --data-dir evaluation/card_game/card_jsd_data \\
        --output-dir plots/card_game
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (npz key, legend label, colour) — colours match plot_ablation_bars.py (tab10).
CONDS = [
    ("ippo",        "IPPO",        "#7D5A3C"),
    ("op_only",     "OP",          "#1f77b4"),
    ("he",          "HE",          "#e377c2"),
    ("mate_noop",   "MATE$-$OP",   "#ff7f0e"),
    ("mate_nofeed", "MATE$-$feed", "#9467bd"),
    ("mate_noaux",  "MATE$-$aux",  "#2ca02c"),
    ("mate",        "MATE",        "#d62728"),
]
YLABEL = "Card-attention JSD"


def _mean_sem(jsd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.sum(~np.isnan(jsd), axis=0)
    return np.nanmean(jsd, axis=0), np.nanstd(jsd, axis=0) / np.sqrt(np.maximum(n, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="evaluation/card_game/card_jsd_data")
    parser.add_argument("--output-dir", default="plots/card_game")
    parser.add_argument("--legend-loc", default="center left")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    series = []
    for key, label, color in CONDS:
        path = data_dir / f"{key}.npz"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        jsd = np.asarray(np.load(path, allow_pickle=True)["jsd"], dtype=np.float64)
        series.append((label, color, *_mean_sem(jsd)))

    n_steps = len(series[0][2])
    top = max(float(np.nanmax(m + s)) for _, _, m, s in series)
    ymax = top + 0.05

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.grid(True, color="0.88", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("0.55")
        sp.set_linewidth(1.0)
    ax.tick_params(labelsize=13, color="0.55")
    ax.set_ylabel(YLABEL, fontsize=15)

    steps = np.arange(1, n_steps + 1)
    for label, color, mean, sem in series:
        ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.18,
                        lw=0, zorder=2)
        ax.plot(steps, mean, color=color, lw=2.2, marker="o", ms=6,
                mfc=color, mec=color, zorder=4, label=label)

    ax.set_xticks(steps)
    ax.set_xlim(0.7, n_steps + 0.3)
    ax.set_ylim(0.0, ymax)
    # Legend outside the axes on the right: 7 series overlap too much to seat it
    # inside without covering a line.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=12,
              frameon=True, framealpha=0.95, edgecolor="0.7", handlelength=1.5,
              handletextpad=0.5, labelspacing=0.5, borderpad=0.7)

    fig.tight_layout()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "card_jsd_all.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
