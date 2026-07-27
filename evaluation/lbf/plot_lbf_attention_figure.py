"""Single combined LBF attention figure, mirroring the card-game figure's layout:
(a) MATE and (b) OP attention filmstrips stacked on the left, a shared colorbar,
and (c) the SP/XP episode-return bars on the right.

The filmstrips are the pre-rendered `muted_gamma0.35` blocks shown via `imshow`;
everything else is drawn natively so it stays vector in the PDF. The OP block's
baked-in colorbar is dropped and a fresh coolwarm bar is drawn spanning both
strips (matching the card figure). The bars are drawn from
`evaluation.lbf.plot_sp_xp_op.COND` at a deliberately landscape aspect so panel
(c) reads compact next to the maps rather than dominating them.

Usage:
    uv run python -m evaluation.lbf.plot_lbf_attention_figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

from evaluation.lbf.plot_sp_xp_op import COND, SP_HATCH

_BLOCK_DIR = Path("plots/lbf/final/muted_gamma0.35")

# Panel labels in the paper's serif body font, matching the card-game figure.
_TIMES = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
if _TIMES.exists():
    font_manager.fontManager.addfont(str(_TIMES))
LABEL_FONT = ["Times New Roman", "serif"]

# Inch geometry, designed for placement at ~\textwidth (7in) so points are real.
# The stacked map block is sized to the bar panel's full height so the two
# columns share one vertical band (see `build`).
GRID_LEFT = 0.70   # room for the (a)/(b) row labels
VGAP = 0.05        # between the MATE and OP strips
BOTTOM_M = 0.16
TOP_M = 0.05
CBAR_W = 0.085
CBAR_GAP = 0.05
CBAR_LABEL_ROOM = 0.42  # colorbar tick labels + gap to the bars

# Bar panel: landscape (aspect ~2) so it reads "less tall" beside the maps.
BAR_W = 2.35
BAR_H = 1.15
BAR_GAP = 0.50     # space between the colorbar labels and the y-axis label
LEGEND_ROOM = 0.24      # above the bar axes
XLABEL_ROOM = 0.20      # below the bar axes (IPPO/OP/HE IPPO/MATE)
CAPTION_ROOM = 0.22     # below that for the (c) caption
RIGHT_M = 0.08

LABEL_PT = 8
YLABEL_PT = 7
XTICK_PT = 7
YTICK_PT = 6
LEGEND_PT = 6.5
CBAR_TICK_PT = 6
CBAR_EDGE_LW = 0.4
CBAR_TICK_LEN = 1.6
CAPTION = "(c) SP and XP scores"


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def _draw_bars(ax) -> None:
    labels = [c[0] for c in COND]
    x = np.arange(len(labels))
    w = 0.38
    ekw = dict(ecolor="black", elinewidth=0.5, capsize=1.6, capthick=0.5)
    plt.rcParams["hatch.linewidth"] = 0.7
    for i, (_, base, spm, spe, xpm, xpe) in enumerate(COND):
        ax.bar(x[i] - w / 2, spm, w, color="white", hatch=SP_HATCH,
               edgecolor=base, linewidth=0.45, yerr=spe, error_kw=ekw, zorder=3)
        ax.bar(x[i] + w / 2, xpm, w, color=base, edgecolor="none",
               yerr=xpe, error_kw=ekw, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=XTICK_PT)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylabel("Episode return", fontsize=YLABEL_PT, labelpad=2)
    ax.set_ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    ax.tick_params(axis="y", labelsize=YTICK_PT, width=0.5, length=2.0, pad=1.5)
    ax.tick_params(axis="x", width=0.5, length=2.0, pad=1.5)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        Patch(facecolor="white", hatch=SP_HATCH, edgecolor="#888888", label="Self-play (SP)"),
        Patch(facecolor="#888888", edgecolor="none", label="Cross-play (XP)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=LEGEND_PT, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.3, handletextpad=0.4,
              columnspacing=1.0, borderpad=0.0)


def build(out_stem: Path, block_dir: Path) -> None:
    mate = _load_rgb(block_dir / "lbf_block_MATE.png")
    op = _load_rgb(block_dir / "lbf_block_OP.png")
    op_frames = op[:, : mate.shape[1]]  # drop the baked-in colorbar; frames match MATE

    # The maps are squeezed to the bar CHART height (legend + axes + x labels),
    # so both columns end on one line. The (c) caption gets its own band just
    # below that shared bottom edge. Strip width follows from the height at the
    # source aspect, so nothing distorts.
    chart_h = LEGEND_ROOM + BAR_H + XLABEL_ROOM
    grid_h = (chart_h - VGAP) / 2
    grid_w = grid_h * mate.shape[1] / mate.shape[0]
    op_b = BOTTOM_M + CAPTION_ROOM  # shared bottom: OP strip bottom = x-labels bottom
    mate_b = op_b + grid_h + VGAP
    top = op_b + chart_h

    cbar_left = GRID_LEFT + grid_w + CBAR_GAP
    bar_left = cbar_left + CBAR_W + CBAR_LABEL_ROOM + BAR_GAP

    W = bar_left + BAR_W + RIGHT_M
    H = top + TOP_M
    fig = plt.figure(figsize=(W, H))

    def ax_in(left, bottom, width, height):
        return fig.add_axes([left / W, bottom / H, width / W, height / H])

    for bottom, img in ((mate_b, mate), (op_b, op_frames)):
        ax = ax_in(GRID_LEFT, bottom, grid_w, grid_h)
        ax.imshow(img, aspect="auto", interpolation="antialiased")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # (a)/(b) row labels, right-aligned just left of each strip, vertically centred.
    for bottom, text in ((mate_b, "(a) MATE"), (op_b, "(b) OP")):
        fig.text((GRID_LEFT - 0.06) / W, (bottom + grid_h / 2) / H, text,
                 ha="right", va="center", fontsize=LABEL_PT, family=LABEL_FONT)

    cax = ax_in(cbar_left, op_b, CBAR_W, chart_h)
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="coolwarm")
    cbar = fig.colorbar(sm, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=CBAR_TICK_PT, length=CBAR_TICK_LEN,
                        width=CBAR_EDGE_LW, color="black", direction="out", pad=2)
    cbar.outline.set_linewidth(CBAR_EDGE_LW)
    cbar.outline.set_edgecolor("black")

    # Bar axes sit below the legend room; x labels and the (c) caption fill the
    # rest, so the panel bottom aligns with the OP strip bottom.
    bar_b = top - LEGEND_ROOM - BAR_H
    _draw_bars(ax_in(bar_left, bar_b, BAR_W, BAR_H))

    # Caption in its own band just below the shared bottom edge of the maps and bars.
    fig.text((bar_left + BAR_W / 2) / W, (BOTTOM_M + CAPTION_ROOM / 2) / H,
             CAPTION, ha="center", va="center", fontsize=LABEL_PT, family=LABEL_FONT)

    # Crop to a uniform margin around visible ink, as in the card-game figure.
    margin = 0.04
    fig.set_dpi(600)  # detect ink at the save resolution so the crop is exact
    fig.canvas.draw()
    ink = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].min(axis=2) < 245
    ph = ink.shape[0]
    cols, rows = np.where(ink.any(axis=0))[0], np.where(ink.any(axis=1))[0]
    d = fig.dpi
    bbox = Bbox([
        [cols.min() / d - margin, (ph - 1 - rows.max()) / d - margin],
        [cols.max() / d + margin, (ph - 1 - rows.min()) / d + margin],
    ])

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=600, bbox_inches=bbox)
        print(f"saved {out_stem.with_suffix(f'.{ext}')}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block-dir", type=Path, default=_BLOCK_DIR)
    p.add_argument("--out", type=Path, default=Path("plots/lbf/lbf_attention_figure"))
    args = p.parse_args()
    build(args.out, args.block_dir)


if __name__ == "__main__":
    main()
