"""Named layouts for the Stag Hunt environment.

A layout is a plain dict (row-major flat indices, `idx = y * width + x`):
  height, width   grid dimensions
  wall_idx        cells that are walls (the border is NOT auto-walled; pass it here
                  if wanted, but the original env uses open rooms)

Unlike Dual Destination, agent / stag / plant positions are NOT part of the
layout: they are sampled randomly on free cells at every `reset` (the original
StagHunt regenerates each episode), and their counts come from the env
constructor (`num_stags`, `num_plants`).
"""
from __future__ import annotations

from typing import TypedDict

import jax.numpy as jnp


class Layout(TypedDict):
    height: int
    width: int
    wall_idx: list[int]


LAYOUTS: dict[str, Layout] = {
    # Open rooms (no interior walls), matching the EmptyStagHuntEnv configs.
    "empty_7x7": {"height": 7, "width": 7, "wall_idx": []},
    "empty_8x8": {"height": 8, "width": 8, "wall_idx": []},
}


def layout_to_arrays(layout: Layout) -> tuple[int, int, jnp.ndarray]:
    """Expand a layout dict into `(height, width, wall_map)`.

    `wall_map` is a `(height, width)` bool grid.
    """
    h, w = layout["height"], layout["width"]
    wall_idx = jnp.asarray(layout["wall_idx"], dtype=jnp.int32)
    wall_map = jnp.zeros((h * w,), dtype=bool).at[wall_idx].set(True).reshape(h, w)
    return h, w, wall_map
