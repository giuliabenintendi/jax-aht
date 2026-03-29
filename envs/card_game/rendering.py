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

AGENT_0_COLOR = jnp.array([255, 140, 0], dtype=jnp.uint8)    # orange
AGENT_1_COLOR = jnp.array([255, 0, 255], dtype=jnp.uint8)   # magenta
GRAY_COLOR = jnp.array([128, 128, 128], dtype=jnp.uint8)     # hidden card

_EMPTY_TILE = jnp.zeros((TILE_PIXELS, TILE_PIXELS, 3), dtype=jnp.uint8)


def _render_tile(mask, color):
    """Render a tile with a colored shape on black background."""
    return jnp.where(mask[:, :, None], color[None, None, :], _EMPTY_TILE)


def render_card_game(card_permutation: jnp.ndarray, revealed=None) -> jnp.ndarray:
    """Render the card game scene.

    Args:
        card_permutation: (5,) int array — card identity at each position.
        revealed: optional (5,) bool array — which cards are visible.
                  If None, all cards shown. If provided, unrevealed = gray.

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
        if revealed is not None:
            color = jnp.where(revealed[i], color, GRAY_COLOR)
        card_tile = _render_tile(_CARD_MASK, color)
        img = jax.lax.dynamic_update_slice(
            img, card_tile, (TILE_PIXELS, i * TILE_PIXELS, 0)
        )
        return img, None

    img, _ = jax.lax.scan(draw_card, img, jnp.arange(NUM_CARDS))

    return img


def _draw_choice_border(img, card_pos, color, thickness=2):
    """Draw a thick border around a card tile (row=1) to indicate an agent's choice.

    Works on the upscaled image. card_pos is the column index (0-4).
    """
    import numpy as np

    h, w = img.shape[:2]
    tile_h = h // GRID_ROWS
    tile_w = w // GRID_COLS

    y0 = tile_h  # card row = 1
    x0 = card_pos * tile_w

    # Top and bottom edges
    img[y0:y0 + thickness, x0:x0 + tile_w] = color
    img[y0 + tile_h - thickness:y0 + tile_h, x0:x0 + tile_w] = color
    # Left and right edges
    img[y0:y0 + tile_h, x0:x0 + thickness] = color
    img[y0:y0 + tile_h, x0 + tile_w - thickness:x0 + tile_w] = color

    return img


def render_card_game_eval_frames(ep_states, scale: int = 32):
    """Render upscaled RGB frames from a list of episode WrappedEnvStates.

    On the decision step (when agent_choices != -1), draws colored borders
    around the chosen cards: orange for agent 0, magenta for agent 1.

    Args:
        ep_states: list of WrappedEnvState (from run_episode_with_states).
        scale: upscale factor (nearest-neighbor) for video quality.

    Returns:
        list of (H_scaled, W_scaled, 3) uint8 numpy arrays.
    """
    import numpy as np
    from PIL import Image

    agent0_color = np.array([255, 140, 0], dtype=np.uint8)    # orange
    agent1_color = np.array([255, 0, 255], dtype=np.uint8)    # magenta
    border_thickness = max(2, scale // 8)

    frames = []
    for state in ep_states:
        img = render_card_game(state.env_state.card_permutation)
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        pil_img = Image.fromarray(img_np).resize(
            (w * scale, h * scale), Image.NEAREST
        )
        frame = np.array(pil_img)

        choices = np.array(state.env_state.agent_choices)
        perm = np.array(state.env_state.card_permutation)
        if choices[0] >= 0:
            # Find position of chosen color
            pos_0 = int(np.where(perm == choices[0])[0][0])
            frame = _draw_choice_border(frame, pos_0, agent0_color, border_thickness)
        if choices[1] >= 0:
            pos_1 = int(np.where(perm == choices[1])[0][0])
            frame = _draw_choice_border(frame, pos_1, agent1_color, border_thickness)

        frames.append(frame)
    return frames
