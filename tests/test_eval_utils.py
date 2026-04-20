import numpy as np

from marl.eval_utils import _draw_message_on_cell


def _find_dot_column(cell, scale):
    tile_w = scale * 7
    mask = np.any(cell != 0, axis=-1)
    ys, xs = np.where(mask)
    assert len(xs) > 0
    return int(xs.min() // tile_w)


def test_draw_message_on_cell_resolves_message_color_to_physical_column():
    scale = 2
    cell = np.zeros((3 * scale * 7, 5 * scale * 7, 3), dtype=np.uint8)
    permutation = np.array([2, 4, 1, 0, 3], dtype=np.int32)

    _draw_message_on_cell(
        cell,
        msg_value=2,
        scale=scale,
        color=[255, 0, 255],
        card_permutation=permutation,
    )

    assert _find_dot_column(cell, scale) == 0


def test_draw_message_on_cell_uses_raw_column_when_no_permutation_is_given():
    scale = 2
    cell = np.zeros((3 * scale * 7, 5 * scale * 7, 3), dtype=np.uint8)

    _draw_message_on_cell(
        cell,
        msg_value=2,
        scale=scale,
        color=[255, 0, 255],
    )

    assert _find_dot_column(cell, scale) == 2
