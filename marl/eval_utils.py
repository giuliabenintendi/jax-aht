"""Generic drawing utilities for eval visualization (card-game borders, boxes)."""


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


def _draw_message_on_cell(cell, msg_pos, scale, color=None):
    """Draw a border around the card at msg_pos (row 1) in the partner's color."""
    tile_h = scale * 7
    tile_w = scale * 7
    frame_h, frame_w = cell.shape[:2]
    grid_rows = frame_h // tile_h
    grid_cols = frame_w // tile_w
    thickness = max(2, scale // 8)
    if color is None:
        color = [139, 90, 43]  # fallback brown
    _draw_box(cell, 1, msg_pos, tile_h, tile_w, color, thickness)
