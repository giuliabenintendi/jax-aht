"""JAX tile-based renderer for the Card Game environment.

Produces a (3*TILE_PIXELS, 5*TILE_PIXELS, 3) uint8 RGB image:
  Row 0: empty | empty | agent_0 (▽) | empty | empty
  Row 1: card  | card  | card         | card  | card
  Row 2: empty | empty | agent_1 (△) | empty | empty

Cards are colored rectangles (5 maximally distinct colors).
Agents are triangles pointing toward the cards.
"""
import jax
import jax.numpy as jnp

TILE_PIXELS = 7

GRID_ROWS = 3
GRID_COLS = 5
NUM_CARDS = 5

# Pixel coordinate grids for mask definitions
_Y, _X = jnp.meshgrid(
    jnp.arange(TILE_PIXELS), jnp.arange(TILE_PIXELS), indexing="ij"
)

# Card: 5×5 filled rectangle (rows 1-5, cols 1-5)
_CARD_MASK = (_Y >= 1) & (_Y <= 5) & (_X >= 1) & (_X <= 5)

# Down-pointing triangle (agent 0, top — points toward cards below)
#   . . . . . . .
#   . X X X X X .
#   . X X X X X .
#   . . X X X . .
#   . . X X X . .
#   . . . X . . .
#   . . . . . . .
_DOWN_TRI_MASK = (
    ((_Y == 1) & (_X >= 1) & (_X <= 5))
    | ((_Y == 2) & (_X >= 1) & (_X <= 5))
    | ((_Y == 3) & (_X >= 2) & (_X <= 4))
    | ((_Y == 4) & (_X >= 2) & (_X <= 4))
    | ((_Y == 5) & (_X == 3))
)

# Up-pointing triangle (agent 1, bottom — points toward cards above)
#   . . . . . . .
#   . . . X . . .
#   . . X X X . .
#   . . X X X . .
#   . X X X X X .
#   . X X X X X .
#   . . . . . . .
_UP_TRI_MASK = (
    ((_Y == 1) & (_X == 3))
    | ((_Y == 2) & (_X >= 2) & (_X <= 4))
    | ((_Y == 3) & (_X >= 2) & (_X <= 4))
    | ((_Y == 4) & (_X >= 1) & (_X <= 5))
    | ((_Y == 5) & (_X >= 1) & (_X <= 5))
)

# 5 maximally distinct card colors
CARD_COLORS = jnp.array([
    [220, 50, 50],    # red
    [50, 100, 220],   # blue
    [50, 180, 50],    # green
    [220, 200, 50],   # yellow
    [160, 50, 200],   # purple
], dtype=jnp.uint8)

# Agent colors (pastel, distinct from saturated card colors)
AGENT_0_COLOR = jnp.array([230, 130, 130], dtype=jnp.uint8)  # salmon
AGENT_1_COLOR = jnp.array([130, 130, 230], dtype=jnp.uint8)  # periwinkle

_EMPTY_TILE = jnp.zeros((TILE_PIXELS, TILE_PIXELS, 3), dtype=jnp.uint8)


def _render_tile(mask, color):
    """Render a tile with a colored shape on black background."""
    return jnp.where(mask[:, :, None], color[None, None, :], _EMPTY_TILE)


def render_card_game(card_permutation: jnp.ndarray) -> jnp.ndarray:
    """Render the card game scene.

    Args:
        card_permutation: (5,) int array — card identity at each position.

    Returns:
        (GRID_ROWS * TILE_PIXELS, GRID_COLS * TILE_PIXELS, 3) uint8 RGB image.
    """
    h_px = GRID_ROWS * TILE_PIXELS
    w_px = GRID_COLS * TILE_PIXELS
    img = jnp.zeros((h_px, w_px, 3), dtype=jnp.uint8)

    # Agent 0 at grid (0, 2) — top center, down-pointing
    agent0_tile = _render_tile(_DOWN_TRI_MASK, AGENT_0_COLOR)
    img = jax.lax.dynamic_update_slice(img, agent0_tile, (0, 2 * TILE_PIXELS, 0))

    # Agent 1 at grid (2, 2) — bottom center, up-pointing
    agent1_tile = _render_tile(_UP_TRI_MASK, AGENT_1_COLOR)
    img = jax.lax.dynamic_update_slice(
        img, agent1_tile, (2 * TILE_PIXELS, 2 * TILE_PIXELS, 0)
    )

    # Cards at grid row 1, columns 0-4
    def draw_card(img, i):
        card_id = card_permutation[i]
        color = CARD_COLORS[card_id]
        card_tile = _render_tile(_CARD_MASK, color)
        img = jax.lax.dynamic_update_slice(
            img, card_tile, (TILE_PIXELS, i * TILE_PIXELS, 0)
        )
        return img, None

    img, _ = jax.lax.scan(draw_card, img, jnp.arange(NUM_CARDS))

    return img
