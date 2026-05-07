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


def _draw_choice_on_cell(cell, choice_pos, agent_idx, scale, card_row=1, card_col=None,
                         color=None):
    """Draw a thick border that hugs the chosen card's 5w x 7h rectangle in
    the minimal cards-only layout (no agent-tile border anymore: agents are
    no longer drawn). `agent_idx` and `card_row` are kept in the signature
    for call-site compatibility but `card_row` is unused under the new
    layout."""
    from envs.card_game.rendering import (
        CARD_RECT_W, CARD_RECT_H, CARD_RECT_Y,
    )

    del agent_idx  # unused — agent tiles are no longer rendered
    del card_row   # unused — card row is fixed by CARD_RECT_Y

    # Decision-step border is intentionally very thick so the picked card
    # is unambiguously highlighted in the eval-video frame.
    thickness = 3 * scale
    if color is None:
        color = [255, 255, 255]
    color_arr = np.asarray(color, dtype=np.uint8)

    col = card_col if card_col is not None else choice_pos
    x0 = (1 + col * (CARD_RECT_W + 2)) * scale
    y0 = int(CARD_RECT_Y) * scale
    w = CARD_RECT_W * scale
    h = CARD_RECT_H * scale

    cell[y0:y0 + thickness, x0:x0 + w] = color_arr
    cell[y0 + h - thickness:y0 + h, x0:x0 + w] = color_arr
    cell[y0:y0 + h, x0:x0 + thickness] = color_arr
    cell[y0:y0 + h, x0 + w - thickness:x0 + w] = color_arr


def _draw_message_on_cell(cell, msg_value, scale, color=None, card_permutation=None):
    """Draw a 2x2 raw-pixel dot at the centre of the messaged card under the
    minimal cards-only layout (5w x 7h cards at CARD_RECT_Y).
    """
    from envs.card_game.rendering import (
        CARD_RECT_W, CARD_RECT_H, CARD_RECT_Y,
    )

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

    dot_size = 2  # raw pixels at obs level
    card_x_origin = 1 + msg_col * (CARD_RECT_W + 2)
    raw_y = int(CARD_RECT_Y) + (CARD_RECT_H - dot_size) // 2
    raw_x = card_x_origin + (CARD_RECT_W - dot_size) // 2
    y0 = raw_y * scale
    x0 = raw_x * scale
    h = dot_size * scale
    w = dot_size * scale
    cell[y0:y0 + h, x0:x0 + w] = color


