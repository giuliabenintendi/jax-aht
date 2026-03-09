"""JAX tile-based renderer for Level Based Foraging.

Produces (grid_size*TILE_PIXELS, grid_size*TILE_PIXELS, 3) uint8 RGB images
from Jumanji LBF State. Matches the visual language of the original LBF
renderer (semitable/lb-foraging): red apples for food, blue agents.

Level is encoded as color intensity — higher level = more saturated color.
This is CNN-friendly and avoids legibility issues with tiny digit fonts.

Entity types:
  - Empty cell: white with thin grey grid lines
  - Agent: blue circle (per-agent hue), intensity encodes level
  - Food: red circle, intensity encodes level
  - Loading agent: orange circle, intensity encodes level
"""
import jax
import jax.numpy as jnp

TILE_PIXELS = 7

# Maximum supported level for intensity scaling
_MAX_LEVEL = 5

# Colors (RGB uint8)
_GRID_LINE = jnp.array([200, 200, 200], dtype=jnp.uint8)

# Base agent colors (fully saturated = max level).
# Per-agent hues to distinguish agents in attention visualizations.
_AGENT_BASE_COLORS = jnp.array([
    [30, 90, 220],    # blue  (agent 0)
    [50, 160, 210],   # teal  (agent 1)
    [100, 80, 200],   # purple (agent 2)
    [20, 140, 120],   # dark teal (agent 3)
], dtype=jnp.float32)

# Food: red, matching the apple from the original renderer
_FOOD_BASE_COLOR = jnp.array([210, 40, 40], dtype=jnp.float32)

# Loading: orange to indicate the load action
_LOADING_BASE_COLOR = jnp.array([230, 130, 20], dtype=jnp.float32)


def _make_empty_tile():
    """White tile with grey grid lines on top and left edges."""
    tile = jnp.full((TILE_PIXELS, TILE_PIXELS, 3), 255, dtype=jnp.uint8)
    tile = tile.at[0, :, :].set(_GRID_LINE)
    tile = tile.at[:, 0, :].set(_GRID_LINE)
    return tile


def _circle_mask():
    """Boolean mask for a circle centered in a TILE_PIXELS tile."""
    y, x = jnp.meshgrid(
        jnp.arange(TILE_PIXELS), jnp.arange(TILE_PIXELS), indexing="ij",
    )
    cy, cx = TILE_PIXELS / 2.0, TILE_PIXELS / 2.0
    r = TILE_PIXELS / 2.0 - 0.8
    return ((y - cy + 0.5) ** 2 + (x - cx + 0.5) ** 2) <= r ** 2


_EMPTY_TILE = _make_empty_tile()
_CIRCLE = _circle_mask()


def _level_color(base_color, level):
    """Blend base_color toward white based on level.

    level=1 (lowest) → washed out (closer to white)
    level=max        → fully saturated base color

    Interpolation: color = white * (1 - t) + base * t
    where t = level / _MAX_LEVEL, clamped to [0.3, 1.0] so even
    level-1 entities are clearly visible.
    """
    t = jnp.clip(level / _MAX_LEVEL, 0.3, 1.0)
    color = 255.0 * (1.0 - t) + base_color * t
    return jnp.clip(color, 0, 255).astype(jnp.uint8)


def _render_entity_tile(color):
    """Render a tile with a colored circle."""
    tile = _EMPTY_TILE.copy()
    tile = jnp.where(_CIRCLE[:, :, None], color[None, None, :], tile)
    return tile


def render_lbf_state(state, grid_size, num_agents, num_food):
    """Render an LBF state to an RGB image.

    Args:
        state: Jumanji LBF State (state.agents, state.food_items)
        grid_size: int, the grid dimension
        num_agents: int, number of agents
        num_food: int, number of food items

    Returns:
        (grid_size * TILE_PIXELS, grid_size * TILE_PIXELS, 3) uint8 array
    """
    h_px = grid_size * TILE_PIXELS
    w_px = grid_size * TILE_PIXELS

    # Build empty grid image
    empty_row = jnp.tile(_EMPTY_TILE[None, :, :, :], (grid_size, 1, 1, 1))
    tiles = jnp.tile(empty_row[None, :, :, :, :], (grid_size, 1, 1, 1, 1))
    img = tiles.transpose(0, 2, 1, 3, 4).reshape(h_px, w_px, 3)

    # Draw food items (red, intensity = level)
    def draw_food(img, i):
        row = state.food_items.position[i, 0]
        col = state.food_items.position[i, 1]
        level = state.food_items.level[i]
        eaten = state.food_items.eaten[i]
        color = _level_color(_FOOD_BASE_COLOR, level)
        food_tile = _render_entity_tile(color)
        y = row * TILE_PIXELS
        x = col * TILE_PIXELS
        img = jnp.where(
            eaten,
            img,
            jax.lax.dynamic_update_slice(img, food_tile, (y, x, 0)),
        )
        return img, None

    img, _ = jax.lax.scan(draw_food, img, jnp.arange(num_food))

    # Draw agents on top (blue per-agent hue, orange if loading)
    def draw_agent(img, i):
        row = state.agents.position[i, 0]
        col = state.agents.position[i, 1]
        level = state.agents.level[i]
        loading = state.agents.loading[i]
        agent_base = _AGENT_BASE_COLORS[i % _AGENT_BASE_COLORS.shape[0]]
        base = jnp.where(loading, _LOADING_BASE_COLOR, agent_base)
        color = _level_color(base, level)
        agent_tile = _render_entity_tile(color)
        y = row * TILE_PIXELS
        x = col * TILE_PIXELS
        img = jax.lax.dynamic_update_slice(img, agent_tile, (y, x, 0))
        return img, None

    img, _ = jax.lax.scan(draw_agent, img, jnp.arange(num_agents))

    return img
