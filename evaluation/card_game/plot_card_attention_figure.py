"""Single combined card-game attention figure: (a) MATE + (b) OP filmstrips and
(c) card-attention JSD, rendered in one matplotlib canvas.

Everything except the card grids (which are inherently raster and shown via
`imshow` of the cropped source filmstrips) is drawn natively, so all text —
the (a)/(b)/(c) labels, the colorbar ticks, and the JSD axes/legend — comes
from one place and stays vector in the PDF. Font sizes are the knobs below.

The card grids come from the two per-episode filmstrips
(`evaluation.card_game.eval_filmstrips`) with their per-strip colorbars cropped
off; MATE is the top block, OP the bottom, matching the caption's (a)/(b). The
JSD curves are re-plotted from `eval_card_jsd.py` npz output in the same house
style as `plot_card_jsd.py`.

Usage:
    uv run python -m evaluation.card_game.plot_card_attention_figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.transforms import Bbox

from evaluation.card_game.combine_filmstrips import _grid_right

_PAPER_DIR = Path(
    "/Users/giuliabenintendi/3. Resources/jax-aht/papers/AAAI_2027/"
    "figures/card-game/filmstrips/final"
)
_JSD_DIR = Path(__file__).parent / "card_jsd_data"

OP_COLOR = "#3b75af"
MATE_COLOR = "#c53a32"

# Panel labels (a)/(b)/(c) in the paper's serif body font. Register the system
# Times New Roman so it resolves by name; fall back to a generic serif elsewhere.
_TIMES = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
if _TIMES.exists():
    font_manager.fontManager.addfont(str(_TIMES))
LABEL_FONT = ["Times New Roman", "serif"]

# Point sizes are real when the figure is placed at width=\linewidth in a
# \textwidth figure* (7.0in). The colorbar mirrors the LBF one: a thin bar with
# no tick marks; its numbers share the JSD tick size so the two panels read alike.
# It runs the full filmstrip height so its 1.00 aligns with the JSD 0.6.
FIG_W_IN = 7.0  # designed for \textwidth; height is derived from the grid aspect
LABEL_PT = 7
JSD_TICK_PT = 6
CBAR_TICK_PT = JSD_TICK_PT
CBAR_W_IN = 0.078  # thin bar, matching the LBF colorbar aspect (~11:1 h/w)
LEGEND_PT = 6
YMAX = 0.6

# Slice of coolwarm that [0, 1] spans. The source strips are painted with the FULL
# map, so `_remap_cmap` moves them onto this slice; bar and cards therefore always
# agree. Full range: muting the top end costs the OP block its deep-red cards, which
# are the whole point of the panel.
CMAP_LO, CMAP_HI = 0.0, 1.0

# Frame/ticks on the colorbar. Weights are tiny because the bar is only
# CBAR_W_IN * 72 ~= 5.6 pt wide — the JSD panel's 0.6 pt spine reads as a hairline on
# a 1.45 in box but as a girder here, eating ~20% of the bar.
CBAR_EDGE_LW = 0.25
CBAR_TICK_LEN = 1.4


def _load_jsd(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=True)["jsd"], dtype=np.float64)


def _mean_sem(jsd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.sum(~np.isnan(jsd), axis=0)
    return np.nanmean(jsd, axis=0), np.nanstd(jsd, axis=0) / np.sqrt(np.maximum(n, 1))


def _slice_cmap(lo: float, hi: float):
    """Coolwarm restricted to [lo, hi], renormalised so it spans 0..1."""
    return LinearSegmentedColormap.from_list(
        "coolwarm_slice", colormaps["coolwarm"](np.linspace(lo, hi, 256)))


def _remap_cmap(strip: np.ndarray, lo: float, hi: float,
                tol: float = 3.0) -> np.ndarray:
    """Recolour a strip's card pixels from full coolwarm onto the [lo, hi] slice.

    `marl.eval_card_game` paints each card as a straight `coolwarm(a)` lookup at
    full intensity — no opacity ramp, no blending — so the mapping inverts exactly:
    a pixel sitting on the coolwarm curve is card attention, and its `a` is the
    curve position. The chrome (black backdrop, white pixel-art digits, the A0/A1
    markers) lies far off the curve and is left untouched, so recolouring here is
    equivalent to re-rendering the strip with the sliced map — and keeps the figure
    reproducible from the existing PNGs without a GPU rollout.
    """
    if (lo, hi) == (0.0, 1.0):
        return strip
    ramp = np.linspace(0.0, 1.0, 1024)
    full = np.asarray(colormaps["coolwarm"](ramp))[:, :3] * 255.0
    sliced = np.asarray(colormaps["coolwarm"](lo + ramp * (hi - lo)))[:, :3] * 255.0

    flat = strip.reshape(-1, 3)
    # Match once per distinct colour: the strips hold a few thousand, not millions.
    key = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) \
        | flat[:, 2].astype(np.int32)
    uniq_key, inverse = np.unique(key, return_inverse=True)
    uniq = np.stack([(uniq_key >> 16) & 255, (uniq_key >> 8) & 255,
                     uniq_key & 255], axis=1).astype(np.float64)
    d = np.linalg.norm(uniq[:, None, :] - full[None, :, :], axis=2)
    idx, res = d.argmin(1), d.min(1)
    out_uniq = uniq.copy()
    on_curve = res < tol
    out_uniq[on_curve] = sliced[idx[on_curve]]
    return out_uniq[inverse].reshape(strip.shape).round().astype(np.uint8)


def _crop_grid(path: Path) -> np.ndarray:
    strip = np.array(plt.imread(path)[:, :, :3])
    if strip.dtype != np.uint8:
        strip = (strip * 255).astype(np.uint8)
    strip = strip[:, : _grid_right(strip)]
    return _remap_cmap(strip, CMAP_LO, CMAP_HI)


def _draw_jsd(ax, jsd: np.ndarray, color: str, label: str | None) -> None:
    mean, sem = _mean_sem(jsd)
    steps = np.arange(1, len(mean) + 1)
    ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.22, lw=0, zorder=2)
    ax.plot(steps, mean, color=color, lw=1.4, marker="o", ms=3, mfc=color, mec=color,
            zorder=4, label=label)


def build(out_stem: Path, mate_strip: Path, op_strip: Path) -> None:
    mate_grid = _crop_grid(mate_strip)
    op_grid = _crop_grid(op_strip)
    op_jsd = _load_jsd(_JSD_DIR / "op_only.npz")
    mate_jsd = _load_jsd(_JSD_DIR / "mate.npz")
    n_steps = op_jsd.shape[1]

    # Explicit inch geometry. The card grids drive their own aspect. The JSD
    # box top aligns with the filmstrip top; its x-ticks and (c) caption occupy
    # `jsd_text_room` at the bottom so the JSD text ends at the filmstrip bottom
    # (both bounded by [op_b, top]) with no empty band below either.
    grid_left, grid_w = 0.80, 3.75
    grid_h = grid_w * mate_grid.shape[0] / mate_grid.shape[1]
    vgap = 0.035
    bottom_margin, top_margin = 0.05, 0.05
    jsd_text_room = 0.28
    block_h = 2 * grid_h + vgap
    op_b = bottom_margin
    mate_b = op_b + grid_h + vgap
    top = op_b + block_h

    W, H = FIG_W_IN, top + top_margin
    fig = plt.figure(figsize=(W, H))

    def ax_in(left, bottom, width, height):
        return fig.add_axes([left / W, bottom / H, width / W, height / H])

    ax_mate = ax_in(grid_left, mate_b, grid_w, grid_h)
    ax_op = ax_in(grid_left, op_b, grid_w, grid_h)
    for ax, grid in ((ax_mate, mate_grid), (ax_op, op_grid)):
        ax.imshow(grid, aspect="auto", interpolation="antialiased")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # (a)/(b) labels right-aligned just left of each block, vertically centred on it.
    for ax, text in ((ax_mate, "(a) MATE"), (ax_op, "(b) OP")):
        box = ax.get_position()
        fig.text(box.x0 - 0.008, (box.y0 + box.y1) / 2, text, ha="right", va="center",
                 fontsize=LABEL_PT, family=LABEL_FONT)

    cax = ax_in(grid_left + grid_w + 0.06, op_b, CBAR_W_IN, block_h)
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=_slice_cmap(CMAP_LO, CMAP_HI))
    cbar = fig.colorbar(sm, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    # Framed bar with outward tick dashes, as on the LBF colorbar.
    cbar.ax.tick_params(labelsize=CBAR_TICK_PT, length=CBAR_TICK_LEN,
                        width=CBAR_EDGE_LW, color="black", direction="out", pad=2)
    cbar.outline.set_visible(True)
    cbar.outline.set_linewidth(CBAR_EDGE_LW)
    cbar.outline.set_edgecolor("black")

    ax_j = ax_in(5.38, op_b + jsd_text_room, 1.45, block_h - jsd_text_room)
    ax_j.grid(True, axis="y", color="0.93", lw=0.8)
    ax_j.set_axisbelow(True)
    for sp in ax_j.spines.values():
        sp.set_color("0.55")
        sp.set_linewidth(0.6)
    ax_j.tick_params(labelsize=JSD_TICK_PT, color="0.55", width=0.5)
    # short x-tick marks with tight pad so the 1-8 labels sit close to the plot,
    # leaving a clear gap down to the (c) caption at the filmstrip bottom.
    ax_j.tick_params(axis="x", length=2, pad=2)
    _draw_jsd(ax_j, op_jsd, OP_COLOR, "OP")
    _draw_jsd(ax_j, mate_jsd, MATE_COLOR, "MATE")
    leg = ax_j.legend(loc="center", bbox_to_anchor=(0.80, 0.44), fontsize=LEGEND_PT,
                      frameon=True, framealpha=0.95, edgecolor="0.85",
                      handlelength=1.3, borderpad=0.08, labelspacing=0.2,
                      handleheight=1.5, handletextpad=0.5)
    leg.get_frame().set_linewidth(0.6)
    ax_j.set_xticks(np.arange(1, n_steps + 1))
    ax_j.set_xlim(0.7, n_steps + 0.3)
    ax_j.set_ylim(0.0, YMAX)
    ax_j.set_yticks(np.arange(0.0, YMAX + 1e-6, 0.2))

    # (c) caption centred on the JSD axes, sat just below the filmstrip bottom
    # to open a small gap up to the x-tick labels.
    jbox = ax_j.get_position()
    fig.text((jbox.x0 + jbox.x1) / 2, (op_b - 0.035) / H, "(c) Card-attention JSD",
             ha="center", va="bottom", fontsize=LABEL_PT, family=LABEL_FONT)

    # Crop to a uniform margin around the *visible ink* (not matplotlib's text
    # bbox, which pads glyphs asymmetrically) so left and right margins are equal.
    margin = 0.03
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
    p.add_argument("--mate", type=Path, default=_PAPER_DIR / "OPJAshaping_gt_filmstrip_xp_0v1_ep6.png")
    p.add_argument("--op", type=Path, default=_PAPER_DIR / "OPonly_gt_filmstrip_xp_0v1_ep2.png")
    p.add_argument("--out", type=Path, default=Path("plots/card_game/card_attention_figure"))
    args = p.parse_args()
    build(args.out, args.mate, args.op)


if __name__ == "__main__":
    main()
