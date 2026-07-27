"""Compose the paper's combined MATE/OP attention filmstrip in one pass.

Stitches the two per-episode 2-row GT-attention filmstrips (produced by
`evaluation.card_game.eval_filmstrips`) into a single 4-row figure with one
shared colorbar and baked-in `(a) MATE` / `(b) OP` row labels, so the LaTeX
side needs no label minipages.

Each source filmstrip already carries its own colorbar (from
`_render_vertical_colorbar` in `marl.eval_card_game`); that per-strip bar is
cropped off and replaced by a single bar rendered at `--cbar-labelsize`, kept
smaller than the row labels.

Usage:
    uv run python -m evaluation.card_game.combine_filmstrips \\
        --mate OPJAshaping_gt_filmstrip_xp_0v1_ep6.png \\
        --op   OPonly_gt_filmstrip_xp_0v1_ep2.png \\
        --out  combined_op_mate_filmstrip.png
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_PAPER_DIR = Path(
    "/Users/giuliabenintendi/3. Resources/jax-aht/papers/AAAI_2027/"
    "figures/card-game/filmstrips/final"
)


def _grid_right(strip: np.ndarray, min_gap: int = 10) -> int:
    """Column where the card grid ends, i.e. the start of the first wide white
    run separating the grid from the appended colorbar. The grid keeps a black
    background, so its white card digits never span a full column."""
    col_white = strip.mean(axis=(0, 2)) > 245
    run = 0
    for x, w in enumerate(col_white):
        run = run + 1 if w else 0
        if run >= min_gap:
            return x - run + 1
    raise ValueError("no grid/colorbar separator found")


def _render_colorbar(height_px: int, labelsize: int) -> np.ndarray:
    """A vertical coolwarm [0, 1] colorbar, scaled to `height_px` rows.

    Mirrors `marl.eval_card_game._render_vertical_colorbar` but with an
    explicit tick-label size instead of one derived from the bar height.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    dpi = 100
    fig, ax = plt.subplots(figsize=(0.9, height_px / dpi), dpi=dpi)
    ColorbarBase(
        ax,
        cmap="coolwarm",
        norm=Normalize(vmin=0.0, vmax=1.0),
        orientation="vertical",
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    ax.tick_params(labelsize=labelsize)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    new_w = max(1, img.width * height_px // img.height)
    return np.array(img.resize((new_w, height_px), Image.LANCZOS))


def combine(
    mate_path: Path,
    op_path: Path,
    out_path: Path,
    *,
    v_gap: int = 44,
    cbar_gap: int = 48,
    cbar_labelsize: int = 80,
    label_font_px: int = 170,
    labels: tuple[str, str] = ("(a) MATE", "(b) OP"),
) -> Path:
    mate = np.array(Image.open(mate_path).convert("RGB"))
    op = np.array(Image.open(op_path).convert("RGB"))
    if mate.shape != op.shape:
        raise ValueError(f"source filmstrips differ in size: {mate.shape} vs {op.shape}")

    gx = _grid_right(mate)
    mate_grid, op_grid = mate[:, :gx], op[:, :gx]
    block_h, grid_w = mate_grid.shape[:2]

    grid_h = 2 * block_h + v_gap
    grids = np.full((grid_h, grid_w, 3), 255, np.uint8)
    grids[:block_h] = mate_grid
    grids[block_h + v_gap :] = op_grid

    cbar = _render_colorbar(grid_h, cbar_labelsize)
    body_w = grid_w + cbar_gap + cbar.shape[1]
    body = np.full((grid_h, body_w, 3), 255, np.uint8)
    body[:, :grid_w] = grids
    body[:, grid_w + cbar_gap :] = cbar

    font = ImageFont.truetype(_dejavu_sans(), label_font_px)
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_w = max(_text_size(tmp, t, font)[0] for t in labels)
    margin = text_w + 88  # 40 left pad + 48 right pad

    out = Image.new("RGB", (margin + body_w, grid_h), "white")
    out.paste(Image.fromarray(body), (margin, 0))
    draw = ImageDraw.Draw(out)
    centers = (block_h // 2, block_h + v_gap + block_h // 2)
    for text, cy in zip(labels, centers, strict=True):
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text(((margin - tw) / 2 - bb[0], cy - th / 2 - bb[1]), text, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"[card_game] Saved combined filmstrip: {out_path} ({out.width}x{out.height} px)")
    return out_path


def _dejavu_sans() -> str:
    import matplotlib.font_manager as fm

    return fm.findfont("DejaVu Sans")


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mate", default="OPJAshaping_gt_filmstrip_xp_0v1_ep6.png")
    p.add_argument("--op", default="OPonly_gt_filmstrip_xp_0v1_ep2.png")
    p.add_argument("--out", default="combined_op_mate_filmstrip_labeled.png")
    p.add_argument("--dir", type=Path, default=_PAPER_DIR, help="base dir for relative paths")
    p.add_argument("--cbar-labelsize", type=int, default=80)
    args = p.parse_args()

    resolve = lambda s: Path(s) if Path(s).is_absolute() else args.dir / s
    combine(
        resolve(args.mate),
        resolve(args.op),
        resolve(args.out),
        cbar_labelsize=args.cbar_labelsize,
    )


if __name__ == "__main__":
    main()
