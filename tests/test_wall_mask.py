"""Visualize wall masks for each Overcooked layout.

Produces a matplotlib figure per layout showing:
- Left: the rendered grid (from a dummy reset)
- Right: the wall mask (red = interior wall, green = meaningful cell)

Run with: uv run python tests/test_wall_mask.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.overcooked.augmented_layouts import augmented_layouts
from envs.overcooked.overcooked_image_wrapper import OvercookedImageWrapper


def compute_interior_wall_mask(layout):
    """Compute interior wall mask: walls that are NOT adjacent to walkable tiles.

    Returns (wall_map, interior_wall_mask, counter_wall_mask) all as (H, W) bool.
    """
    wall_map = np.array(layout["wall_map"])  # (H, W) bool, True = wall
    walkable = ~wall_map

    padded = np.pad(walkable, 1, constant_values=False)
    adjacent_to_walkable = (
        padded[:-2, 1:-1] | padded[2:, 1:-1] |
        padded[1:-1, :-2] | padded[1:-1, 2:]
    )
    counter_walls = wall_map & adjacent_to_walkable
    interior_walls = wall_map & ~counter_walls

    return wall_map, interior_walls, counter_walls


def main():
    layouts_to_test = ["cramped_room", "coord_ring", "counter_circuit",
                       "asymm_advantages", "forced_coord"]

    for name in layouts_to_test:
        if name not in augmented_layouts:
            print(f"[skip] layout '{name}' not found")
            continue

        layout = augmented_layouts[name]
        wall_map, interior_walls, counter_walls = compute_interior_wall_mask(layout)
        walkable = ~np.array(wall_map)
        h, w = wall_map.shape

        # Render the grid
        env = OvercookedImageWrapper(layout=layout, max_steps=400)
        _, state = env.reset(jax.random.PRNGKey(0))
        from envs.overcooked.rendering import render_state
        frame = np.array(render_state(state.env_state))

        # Build color-coded mask visualization
        vis = np.ones((h, w, 3), dtype=np.float32) * 0.5  # grey default
        vis[walkable] = [0.0, 0.8, 0.0]       # green = walkable
        vis[counter_walls] = [0.0, 0.4, 0.8]   # blue = counter/pot walls
        vis[interior_walls] = [0.9, 0.1, 0.1]  # red = interior walls (to mask)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(frame)
        axes[0].set_title(f"{name} — rendered")
        axes[0].axis("off")

        im = axes[1].imshow(vis, interpolation="nearest")
        axes[1].set_title(f"{name} — wall mask")
        for r in range(h):
            for c in range(w):
                label = "W" if interior_walls[r, c] else ("C" if counter_walls[r, c] else ".")
                color = "white" if interior_walls[r, c] else "black"
                axes[1].text(c, r, label, ha="center", va="center",
                             fontsize=max(6, 28 // max(h, w)), color=color, fontweight="bold")
        axes[1].set_xticks(np.arange(-0.5, w, 1), minor=True)
        axes[1].set_yticks(np.arange(-0.5, h, 1), minor=True)
        axes[1].grid(which="minor", color="black", linewidth=1)
        axes[1].tick_params(which="minor", size=0)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        out_path = f"results/wall_mask_{name}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        n_interior = int(interior_walls.sum())
        n_counter = int(counter_walls.sum())
        n_walkable = int(walkable.sum())
        print(f"[{name}] {h}x{w} grid: {n_walkable} walkable (.), "
              f"{n_counter} counter (C), {n_interior} interior wall (W)")
        print(f"  -> saved {out_path}")


if __name__ == "__main__":
    main()
