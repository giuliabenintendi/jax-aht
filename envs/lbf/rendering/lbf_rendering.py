"""JAX tile-based renderer for Level Based Foraging.

Produces (grid_size*TILE_PIXELS, grid_size*TILE_PIXELS, 3) uint8 RGB images
from Jumanji LBF State. Mirrors the original LBF renderer style:
black background, white agents (rectangles), red food (circles).

Shape masks are defined in pixel coordinates centered at pixel (3,3).
Black background provides cell separation (no explicit grid lines needed
at 7px resolution, matching the Overcooked renderer behavior).

Level is encoded as color intensity — higher level = more saturated.
"""
import jax
import jax.numpy as jnp

TILE_PIXELS = 7

_AGENT_BASE_COLORS = jnp.array([
    [255, 255, 255],  # white
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
], dtype=jnp.float32)

_FOOD_BASE_COLOR = jnp.array([210, 40, 40], dtype=jnp.float32)
_LOADING_BASE_COLOR = jnp.array([230, 130, 20], dtype=jnp.float32)


# --- Pixel-space shape masks (centered at pixel 3,3 = true center of 7x7) ---

_Y, _X = jnp.meshgrid(jnp.arange(TILE_PIXELS), jnp.arange(TILE_PIXELS), indexing="ij")

# Agent: 5x5 rectangle (pixels 1-5), centered at pixel 3
_RECT_MASK = (_Y >= 1) & (_Y <= 5) & (_X >= 1) & (_X <= 5)

# Food: circle centered at pixel (3,3) with r²=5
# Gives a rounded shape (21 pixels) with cut corners, distinct from the 5x5 rect
_CIRCLE_MASK = ((_X - 3) ** 2 + (_Y - 3) ** 2) <= 5

# Pure black empty tile
_EMPTY_TILE = jnp.zeros((TILE_PIXELS, TILE_PIXELS, 3), dtype=jnp.uint8)


def _level_color(base_color, level, max_level):
    """Blend base_color toward black based on level.

    Scaled to max_level so level=max gives the full base_color.
    Clamped to [0.3, 1.0] so even level-1 entities are clearly visible.
    """
    t = jnp.clip(level / max_level, 0.3, 1.0)
    color = base_color * t
    return jnp.clip(color, 0, 255).astype(jnp.uint8)


def _render_tile(mask, color):
    """Render a tile with a colored shape on black background."""
    tile = jnp.where(mask[:, :, None], color[None, None, :], _EMPTY_TILE)
    return tile


def render_lbf_state(state, grid_size, num_agents, num_food,
                     max_agent_level=3, max_food_level=6):
    """Render an LBF state to an RGB image.

    Args:
        state: Jumanji LBF State (state.agents, state.food_items)
        grid_size: int, the grid dimension
        num_agents: int, number of agents
        num_food: int, number of food items
        max_agent_level: int, max agent level (for color scaling)
        max_food_level: int, max food level (typically max_agent_level * num_agents)

    Returns:
        (grid_size * TILE_PIXELS, grid_size * TILE_PIXELS, 3) uint8 array
    """
    h_px = grid_size * TILE_PIXELS
    w_px = grid_size * TILE_PIXELS

    # Build empty black image
    img = jnp.zeros((h_px, w_px, 3), dtype=jnp.uint8)

    # Draw food (red circle, intensity = level)
    def draw_food(img, i):
        row = state.food_items.position[i, 0]
        col = state.food_items.position[i, 1]
        level = state.food_items.level[i]
        eaten = state.food_items.eaten[i]
        color = _level_color(_FOOD_BASE_COLOR, level, max_food_level)
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
        color = _level_color(base, level, max_agent_level)
        agent_tile = _render_tile(_RECT_MASK, color)
        y = row * TILE_PIXELS
        x = col * TILE_PIXELS
        img = jax.lax.dynamic_update_slice(img, agent_tile, (y, x, 0))
        return img, None

    img, _ = jax.lax.scan(draw_agent, img, jnp.arange(num_agents))

    return img
