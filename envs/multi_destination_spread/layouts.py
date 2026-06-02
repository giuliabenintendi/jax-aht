"""Named layouts for the Multi-Destination Spread environment.

A layout is a plain dict with flat-index lists (row-major, `idx = y * width + x`):
  height, width   grid dimensions
  num_agents      number of agents (all spawn on `start_idx`)
  wall_idx        cells that are walls
  start_idx       single cell where every agent spawns (the grid centre)
  goal_idx        goal cells (one per goal; the reward counts distinct goals covered)

All agents spawn on the same central cell and the goals are placed symmetrically
around it, so each agent is equidistant from every goal and the assignment of
agents to goals is genuinely ambiguous — the source of the self-play/cross-play
gap. `cardinal_5x5` places the four goals on the edge mid-points (N/S/E/W), two
steps from the centre; `corners_5x5` places them on the four corners (also two
steps away with 8-directional movement). Both are the paper's 5x5 / 4-goal setup.
"""
from __future__ import annotations

from typing import TypedDict

import jax.numpy as jnp


class Layout(TypedDict):
    height: int
    width: int
    num_agents: int
    wall_idx: list[int]
    start_idx: int
    goal_idx: list[int]


LAYOUTS: dict[str, Layout] = {
    # 5x5 empty room, all 4 agents at centre (x=2,y=2), goals on the edge
    # mid-points: (2,0) N, (0,2) W, (4,2) E, (2,4) S. Each is 2 steps from centre.
    "cardinal_5x5": {
        "height": 5,
        "width": 5,
        "num_agents": 4,
        "wall_idx": [],
        "start_idx": 12,            # (x=2, y=2)
        "goal_idx": [2, 10, 14, 22],  # N, W, E, S
    },
    # 5x5 empty room, goals on the four corners: (0,0),(4,0),(0,4),(4,4).
    "corners_5x5": {
        "height": 5,
        "width": 5,
        "num_agents": 4,
        "wall_idx": [],
        "start_idx": 12,
        "goal_idx": [0, 4, 20, 24],
    },
}


def layout_to_arrays(
    layout: Layout,
) -> tuple[int, int, int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Expand a layout dict into `(height, width, num_agents, wall_map, goal_pos, agent_start)`.

    `wall_map` is a `(height, width)` bool grid; `goal_pos` is `(num_goals, 2)` and
    `agent_start` is `(num_agents, 2)`, both int32 `[x, y]` arrays. Every agent
    starts on the single `start_idx` cell.
    """
    h, w = layout["height"], layout["width"]
    n = layout["num_agents"]
    wall_idx = jnp.asarray(layout["wall_idx"], dtype=jnp.int32)
    wall_map = jnp.zeros((h * w,), dtype=bool).at[wall_idx].set(True).reshape(h, w)

    def to_xy(idx_list: list[int]) -> jnp.ndarray:
        idx = jnp.asarray(idx_list, dtype=jnp.int32)
        return jnp.stack([idx % w, idx // w], axis=-1).astype(jnp.int32)

    goal_pos = to_xy(layout["goal_idx"])
    start_xy = to_xy([layout["start_idx"]])[0]
    agent_start = jnp.broadcast_to(start_xy, (n, 2)).astype(jnp.int32)
    return h, w, n, wall_map, goal_pos, agent_start
