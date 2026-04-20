"""Generic drawing utilities for eval visualization (card-game borders, boxes)."""

import numpy as np


def _draw_box(cell, row, col, tile_h, tile_w, color, thickness):
    """Draw a thick border around a tile at grid (row, col) on an upscaled frame."""
    y0 = row * tile_h
    x0 = col * tile_w
    cell[y0:y0 + thickness, x0:x0 + tile_w] = color
    cell[y0 + tile_h - thickness:y0 + tile_h, x0:x0 + tile_w] = color
    cell[y0:y0 + tile_h, x0:x0 + thickness] = color
    cell[y0:y0 + tile_h, x0 + tile_w - thickness:x0 + tile_w] = color


def _draw_choice_on_cell(cell, choice_pos, agent_idx, scale, card_row=1, card_col=None):
    """Draw white borders around the agent tile and its chosen card.

    Derives grid dimensions from the frame shape and TILE_PIXELS.
    Works for both static (3x5) and dynamic (6x10) grids.
    """
    tile_h = scale * 7
    tile_w = scale * 7
    frame_h, frame_w = cell.shape[:2]
    grid_rows = frame_h // tile_h
    grid_cols = frame_w // tile_w
    thickness = max(2, scale // 8)
    white = [255, 255, 255]

    col = card_col if card_col is not None else choice_pos
    _draw_box(cell, card_row, col, tile_h, tile_w, white, thickness)
    agent_row = 0 if agent_idx == 0 else grid_rows - 1
    agent_col = grid_cols // 2
    _draw_box(cell, agent_row, agent_col, tile_h, tile_w, white, thickness)


def _draw_message_on_cell(cell, msg_value, scale, color=None, card_permutation=None):
    """Draw a dot on the messaged card.

    Args:
        cell: Upscaled RGB frame to mutate in place.
        msg_value: Card color/id in the card game message protocol.
        scale: Frame upscale factor.
        color: Dot color.
        card_permutation: Optional mapping from physical column -> card color/id.
            When provided, the helper resolves ``msg_value`` to the card's current
            physical column before drawing. Without it, ``msg_value`` is treated
            as an already-resolved column index for backward compatibility.
    """
    tile_h = scale * 7
    tile_w = scale * 7
    if color is None:
        color = [139, 90, 43]

    if msg_value < 0:
        return

    msg_col = int(msg_value)
    if card_permutation is not None:
        matches = np.where(np.asarray(card_permutation) == msg_value)[0]
        if len(matches) == 0:
            return
        msg_col = int(matches[0])

    # 2x2 dot at center of card tile (row 1), scaled up
    cy = 1 * tile_h + tile_h // 2
    cx = msg_col * tile_w + tile_w // 2
    dot_size = scale * 2  # 2 pixels at obs level, scaled
    cell[cy:cy + dot_size, cx:cx + dot_size] = color


def _draw_decision_square(cell, scale):
    """Draw a white square at top-left to indicate decision step."""
    size = scale * 4  # 4 pixels at obs level, scaled
    cell[0:size, 0:size] = [255, 255, 255]


def _draw_timestep_label(cell, timestep, decision=False):
    """Draw a timestep label on a visualization cell/frame."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    img = Image.fromarray(cell)
    draw = ImageDraw.Draw(img)
    label = f"t={timestep}"
    if decision:
        label += " D"

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=20)
    except OSError:
        font = ImageFont.load_default()

    x0, y0 = 6, 6
    bbox = draw.textbbox((x0, y0), label, font=font)
    pad_x, pad_y = 8, 6
    rect = (
        bbox[0] - pad_x,
        bbox[1] - pad_y,
        bbox[2] + pad_x,
        bbox[3] + pad_y,
    )
    draw.rectangle(rect, fill=(255, 255, 255))
    draw.text((x0, y0), label, fill=(0, 0, 0), font=font, stroke_width=1)
    cell[:] = np.array(img)
