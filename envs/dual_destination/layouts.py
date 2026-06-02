"""Named layouts for the Dual Destination environment.

A layout is a plain dict with flat-index lists (row-major, `idx = y * width + x`):
  height, width   grid dimensions
  wall_idx        cells that are walls
  agent_idx       two agent start cells (agent_0, agent_1)
  goal_idx        two goal cells

`symmetric_5x5` is the intended coordination testbed: an empty room with the two
agents and the two goals placed symmetrically, so each agent is equidistant from
both goals and the goal assignment is genuinely ambiguous. `default` is the small
maze from the original DreamTeam env, useful as a sanity layout.
"""
from __future__ import annotations

from typing import TypedDict

import jax.numpy as jnp


class Layout(TypedDict):
    height: int
    width: int
    wall_idx: list[int]
    agent_idx: list[int]
    goal_idx: list[int]


LAYOUTS: dict[str, Layout] = {
    # 5x5 empty room. Agents at left/right mid-row, goals at top/bottom mid-col.
    # Each agent is Manhattan-distance 4 from both goals -> assignment ambiguous.
    "symmetric_5x5": {
        "height": 5,
        "width": 5,
        "wall_idx": [],
        "agent_idx": [10, 14],  # (x=0,y=2), (x=4,y=2)
        "goal_idx": [2, 22],    # (x=2,y=0), (x=2,y=4)
    },
    # Original DreamTeam 5x5 maze (see dual_destination __main__ example).
    "default": {
        "height": 5,
        "width": 5,
        "wall_idx": [3, 7, 23],   # (x=3,y=0), (x=2,y=1), (x=3,y=4)
        "agent_idx": [0, 24],     # (x=0,y=0), (x=4,y=4)
        "goal_idx": [15, 13],     # (x=0,y=3), (x=3,y=2)
    },
}


def layout_to_arrays(
    layout: Layout,
) -> tuple[int, int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Expand a layout dict into `(height, width, wall_map, goal_pos, agent_start)`.

    `wall_map` is a `(height, width)` bool grid; `goal_pos` and `agent_start` are
    `(2, 2)` int32 arrays of `[x, y]` coordinates.
    """
    h, w = layout["height"], layout["width"]
    wall_idx = jnp.asarray(layout["wall_idx"], dtype=jnp.int32)
    wall_map = jnp.zeros((h * w,), dtype=bool).at[wall_idx].set(True).reshape(h, w)

    def to_xy(idx_list: list[int]) -> jnp.ndarray:
        idx = jnp.asarray(idx_list, dtype=jnp.int32)
        return jnp.stack([idx % w, idx // w], axis=-1).astype(jnp.int32)

    return h, w, wall_map, to_xy(layout["goal_idx"]), to_xy(layout["agent_idx"])
