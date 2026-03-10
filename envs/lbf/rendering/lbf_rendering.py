"""JAX tile-based renderer for Level Based Foraging.

Produces (grid_size*TILE_PIXELS, grid_size*TILE_PIXELS, 3) uint8 RGB images
from Jumanji LBF State. Mirrors the original LBF renderer style:
black background, thin white grid, white agents, red food.

Uses geometric shape functions (same approach as the Overcooked renderer):
  - Empty cell: black with thin white grid lines
  - Agent: white filled rectangle
  - Food: red filled circle
  - Loading agent: orange rectangle

Level is encoded as color intensity — higher level = more saturated.
"""
import jax
import jax.numpy as jnp

TILE_PIXELS = 7

_MAX_LEVEL = 5

# Agent colors (white for all, distinguishable via ego border)
_AGENT_BASE_COLORS = jnp.array([
    [255, 255, 255],  # white (agent 0)
    [255, 255, 255],  # white (agent 1)
    [255, 255, 255],  # white (agent 2)
    [255, 255, 255],  # white (agent 3)
], dtype=jnp.float32)

_FOOD_BASE_COLOR = jnp.array([210, 40, 40], dtype=jnp.float32)
_LOADING_BASE_COLOR = jnp.array([230, 130, 20], dtype=jnp.float32)


# --- Geometric shape functions (Overcooked-style, normalized [0,1] coords) ---

def _make_coord_grid():
    """Normalized coordinate grid for a TILE_PIXELS x TILE_PIXELS tile."""
    y, x = jnp.meshgrid(
        jnp.arange(TILE_PIXELS), jnp.arange(TILE_PIXELS), indexing="ij",
    )
    yf = (y + 0.5) / TILE_PIXELS
    xf = (x + 0.5) / TILE_PIXELS
    return xf, yf


_XF, _YF = _make_coord_grid()


def _point_in_rect(xmin, xmax, ymin, ymax):
    return (_XF >= xmin) & (_XF <= xmax) & (_YF >= ymin) & (_YF <= ymax)


def _point_in_circle(cx, cy, r):
    return (_XF - cx) ** 2 + (_YF - cy) ** 2 <= r ** 2


# --- Precomputed masks ---

# Thin white grid lines: first pixel row (top) and first pixel col (left)
_GRID_MASK = _point_in_rect(0, 1, 0, 1 / TILE_PIXELS) | _point_in_rect(0, 1 / TILE_PIXELS, 0, 1)
_GRID_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)

# Agent: filled rectangle (inner region with margin)
_RECT_MASK = _point_in_rect(0.15, 0.85, 0.15, 0.85)

# Food: filled circle
_CIRCLE_MASK = _point_in_circle(0.5, 0.5, 0.35)


def _make_empty_tile():
    """Black tile with thin white grid lines."""
    tile = jnp.zeros((TILE_PIXELS, TILE_PIXELS, 3), dtype=jnp.uint8)
    tile = jnp.where(_GRID_MASK[:, :, None], _GRID_COLOR[None, None, :], tile)
    return tile


_EMPTY_TILE = _make_empty_tile()

# Keep these exported for tests
_SQUARE = _RECT_MASK
_DIAMOND = _CIRCLE_MASK


def _level_color(base_color, level):
    """Blend base_color toward black based on level.

    level=1 (lowest) -> dimmer (closer to black)
    level=max        -> fully saturated base color

    Clamped to [0.3, 1.0] so even level-1 entities are clearly visible.
    """
    t = jnp.clip(level / _MAX_LEVEL, 0.3, 1.0)
    color = base_color * t
    return jnp.clip(color, 0, 255).astype(jnp.uint8)


def _render_tile(mask, color):
    """Render a tile with a colored shape defined by mask."""
    tile = _EMPTY_TILE.copy()
    tile = jnp.where(mask[:, :, None], color[None, None, :], tile)
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

    # Build empty grid image (black with white grid lines)
    empty_row = jnp.tile(_EMPTY_TILE[None, :, :, :], (grid_size, 1, 1, 1))
    tiles = jnp.tile(empty_row[None, :, :, :, :], (grid_size, 1, 1, 1, 1))
    img = tiles.transpose(0, 2, 1, 3, 4).reshape(h_px, w_px, 3)

    # Draw food (red circle, intensity = level)
    def draw_food(img, i):
        row = state.food_items.position[i, 0]
        col = state.food_items.position[i, 1]
        level = state.food_items.level[i]
        eaten = state.food_items.eaten[i]
        color = _level_color(_FOOD_BASE_COLOR, level)
        food_tile = _render_tile(_CIRCLE_MASK, color)
        y = row * TILE_PIXELS
        x = col * TILE_PIXELS
        img = jnp.where(
            eaten,
            img,
            jax.lax.dynamic_update_slice(img, food_tile, (y, x, 0)),
        )
        return img, None

    img, _ = jax.lax.scan(draw_food, img, jnp.arange(num_food))

    # Draw agents (white rectangle, orange if loading)
    def draw_agent(img, i):
        row = state.agents.position[i, 0]
        col = state.agents.position[i, 1]
        level = state.agents.level[i]
        loading = state.agents.loading[i]
        agent_base = _AGENT_BASE_COLORS[i % _AGENT_BASE_COLORS.shape[0]]
        base = jnp.where(loading, _LOADING_BASE_COLOR, agent_base)
        color = _level_color(base, level)
        agent_tile = _render_tile(_RECT_MASK, color)
        y = row * TILE_PIXELS
        x = col * TILE_PIXELS
        img = jax.lax.dynamic_update_slice(img, agent_tile, (y, x, 0))
        return img, None

    img, _ = jax.lax.scan(draw_agent, img, jnp.arange(num_agents))

    return img
