"""Render card-level JSD-over-episode curves from `eval_card_jsd.py` output.

Three figures in one house style (framed axes, light grid, framed legend):
  - `card_jsd_combined.png` — MATE and OP-only on shared axes,
  - `card_jsd_mate.png`, `card_jsd_op.png` — one condition each.

Card-attention JSD (agents' attention over the five cards, compared in the
shared GT-card frame): convergent joint attention pulls the curve toward 0.

Usage:
    python -m evaluation.card_game.plot_card_jsd \\
        --op   card_jsd_data/op_only.npz \\
        --mate card_jsd_data/mate.npz \\
        --output-dir plots/card_game
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OP_COLOR = "#3b75af"      # steel blue
MATE_COLOR = "#c53a32"    # brick red

# The figure is placed in the paper as panel (c) of Figure 4, inside a
# 0.30\linewidth minipage of a figure* (\textwidth = 7.0in), i.e. at 2.10in
# wide. Rendering at that true size with width=\linewidth gives scale 1.0, so
# the point sizes below are exactly what appears in the PDF; they are set near
# the caption/body font (10pt) so the panel text reads at caption size.
# The height is matched to the two-row filmstrip block beside it (~0.86in) so
# the three panels align; the y-axis is left unlabelled since the panel's own
# sub-caption ("(c) Card-attention JSD") already names the quantity.
FIG_W_IN = 2.10
FIG_H_IN = 1.08
TICK_PT = 9
LEGEND_PT = 9
YMAX = 0.6  # OP plateaus at ~0.47; cap below ln2 to use the vertical space


def _load(path: Path) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["jsd"], dtype=np.float64)


def _mean_sem(jsd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.sum(~np.isnan(jsd), axis=0)
    mean = np.nanmean(jsd, axis=0)
    sem = np.nanstd(jsd, axis=0) / np.sqrt(np.maximum(n, 1))
    return mean, sem


def _new_axes():
    # constrained_layout fits the labels inside the fixed canvas, so the saved
    # PDF/PNG stays exactly FIG_W_IN wide (no bbox trimming) and lands at scale 1.
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), layout="constrained")
    ax.grid(True, axis="y", color="0.93", lw=0.8)  # horizontal gridlines only
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("0.55")
        sp.set_linewidth(1.0)
    ax.tick_params(labelsize=TICK_PT, color="0.55")
    return fig, ax


def _draw(ax, jsd: np.ndarray, color: str, label: str | None) -> None:
    mean, sem = _mean_sem(jsd)
    steps = np.arange(1, len(mean) + 1)
    ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.22,
                    lw=0, zorder=2)
    ax.plot(steps, mean, color=color, lw=1.4, marker="o", ms=3,
            mfc=color, mec=color, zorder=4, label=label)


def _finish(fig, ax, n_steps: int, out: Path) -> None:
    ax.set_xticks(np.arange(1, n_steps + 1))
    ax.set_xlim(0.7, n_steps + 0.3)
    ax.set_ylim(0.0, YMAX)
    ax.set_yticks(np.arange(0.0, YMAX + 1e-6, 0.2))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=1200)  # high-res PNG for direct \includegraphics
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--mate", required=True)
    parser.add_argument("--output-dir", default="plots/card_game")
    args = parser.parse_args()

    op = _load(Path(args.op))
    mate = _load(Path(args.mate))
    n_steps = op.shape[1]
    out_dir = Path(args.output_dir)

    fig, ax = _new_axes()
    _draw(ax, op, OP_COLOR, "OP")
    _draw(ax, mate, MATE_COLOR, "MATE")
    # Framed legend, parked in the clear band between the OP plateau (~0.47) and
    # the MATE tail (~0.06) on the right, so it covers neither curve.
    # handleheight=1.5 makes each row handle-driven so the caps sit centred:
    # this cancels the empty descender space that otherwise leaves the bottom
    # padding larger than the top, giving equal top/bottom margins.
    leg = ax.legend(loc="center", bbox_to_anchor=(0.80, 0.44), fontsize=LEGEND_PT,
                    frameon=True, framealpha=0.95, edgecolor="0.85",
                    handlelength=1.3, borderpad=0.08, labelspacing=0.2,
                    handleheight=1.5, handletextpad=0.5)
    leg.get_frame().set_linewidth(0.6)  # faint border, white bg still masks the grid
    _finish(fig, ax, n_steps, out_dir / "card_jsd_combined.png")

    fig, ax = _new_axes()
    _draw(ax, mate, MATE_COLOR, None)
    _finish(fig, ax, n_steps, out_dir / "card_jsd_mate.png")

    fig, ax = _new_axes()
    _draw(ax, op, OP_COLOR, None)
    _finish(fig, ax, n_steps, out_dir / "card_jsd_op.png")


if __name__ == "__main__":
    main()
