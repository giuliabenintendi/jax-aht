"""JAX tile-based renderer for the Card Game environment (dynamic positions).

Produces a (3*TILE_PIXELS, 5*TILE_PIXELS, 3) uint8 RGB image with
cards at dynamic positions on the 3×5 grid. Follows the same pattern
as LBF rendering (jax.lax.scan + dynamic_update_slice + jnp.where).

Agents are fixed at (0,2) and (2,2).
Cards are colored rectangles placed at random grid positions each episode.
"""
import jax
import jax.numpy as jnp

TILE_PIXELS = 7

GRID_ROWS = 6
GRID_COLS = 10
NUM_COLORS = 10

# Pixel coordinate grids for mask definitions
_Y, _X = jnp.meshgrid(
    jnp.arange(TILE_PIXELS), jnp.arange(TILE_PIXELS), indexing="ij"
)

# Card: 5×5 filled rectangle (rows 1-5, cols 1-5)
_CARD_MASK = (_Y >= 1) & (_Y <= 5) & (_X >= 1) & (_X <= 5)

# Down-pointing triangle (agent 0, top — points toward cards below)
_DOWN_TRI_MASK = (
    ((_Y == 1) & (_X >= 1) & (_X <= 5))
    | ((_Y == 2) & (_X >= 1) & (_X <= 5))
    | ((_Y == 3) & (_X >= 2) & (_X <= 4))
    | ((_Y == 4) & (_X >= 2) & (_X <= 4))
    | ((_Y == 5) & (_X == 3))
)

# Up-pointing triangle (agent 1, bottom — points toward cards above)
_UP_TRI_MASK = (
    ((_Y == 1) & (_X == 3))
    | ((_Y == 2) & (_X >= 2) & (_X <= 4))
    | ((_Y == 3) & (_X >= 2) & (_X <= 4))
    | ((_Y == 4) & (_X >= 1) & (_X <= 5))
    | ((_Y == 5) & (_X >= 1) & (_X <= 5))
)

# 5 maximally distinct card colors
CARD_COLORS = jnp.array([
    [230, 25, 25],    # 0: red
    [25, 50, 230],    # 1: blue
    [25, 200, 25],    # 2: green
    [240, 220, 25],   # 3: yellow
    [150, 25, 220],   # 4: purple
    [25, 220, 220],   # 5: cyan
    [255, 150, 200],  # 6: pink
    [140, 100, 50],   # 7: brown
    [255, 255, 255],  # 8: white
    [128, 128, 128],  # 9: gray
], dtype=jnp.uint8)

AGENT_0_COLOR = jnp.array([255, 140, 0], dtype=jnp.uint8)    # orange
AGENT_1_COLOR = jnp.array([255, 0, 255], dtype=jnp.uint8)   # magenta

_EMPTY_TILE = jnp.zeros((TILE_PIXELS, TILE_PIXELS, 3), dtype=jnp.uint8)


def _render_tile(mask, color):
    """Render a tile with a colored shape on black background."""
    return jnp.where(mask[:, :, None], color[None, None, :], _EMPTY_TILE)


def render_card_game(card_positions, card_present):
    """Render the card game scene with dynamic card positions.

    Args:
        card_positions: (NUM_COLORS, 2) int array — [row, col] per color slot.
        card_present: (NUM_COLORS,) bool array — which colors exist.

    Returns:
        (GRID_ROWS * TILE_PIXELS, GRID_COLS * TILE_PIXELS, 3) uint8 RGB image.
    """
    h_px = GRID_ROWS * TILE_PIXELS
    w_px = GRID_COLS * TILE_PIXELS
    img = jnp.zeros((h_px, w_px, 3), dtype=jnp.uint8)

    # Agent 0 at top center, down-pointing
    agent0_tile = _render_tile(_DOWN_TRI_MASK, AGENT_0_COLOR)
    img = jax.lax.dynamic_update_slice(
        img, agent0_tile, (0, (GRID_COLS // 2) * TILE_PIXELS, 0))

    # Agent 1 at bottom center, up-pointing
    agent1_tile = _render_tile(_UP_TRI_MASK, AGENT_1_COLOR)
    img = jax.lax.dynamic_update_slice(
        img, agent1_tile, ((GRID_ROWS - 1) * TILE_PIXELS, (GRID_COLS // 2) * TILE_PIXELS, 0))

    # Cards at dynamic positions (LBF pattern: scan + conditional rendering)
    def draw_card(img, i):
        row = card_positions[i, 0]
        col = card_positions[i, 1]
        present = card_present[i]
        color = CARD_COLORS[i]  # color index = slot index
        card_tile = _render_tile(_CARD_MASK, color)
        y = row * TILE_PIXELS
        x = col * TILE_PIXELS
        img = jnp.where(
            present,
            jax.lax.dynamic_update_slice(img, card_tile, (y, x, 0)),
            img,
        )
        return img, None

    img, _ = jax.lax.scan(draw_card, img, jnp.arange(NUM_COLORS))

    return img


def render_card_game_eval_frames(ep_states, scale: int = 32):
    """Render upscaled RGB frames for eval visualization."""
    import numpy as np
    from PIL import Image

    agent0_color = np.array([255, 140, 0], dtype=np.uint8)
    agent1_color = np.array([255, 0, 255], dtype=np.uint8)
    border_thickness = max(2, scale // 8)

    frames = []
    for state in ep_states:
        es = state.env_state
        img = render_card_game(es.card_positions, es.card_present)
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        pil_img = Image.fromarray(img_np).resize(
            (w * scale, h * scale), Image.NEAREST
        )
        frame = np.array(pil_img)

        choices = np.array(es.agent_choices)
        if choices[0] >= 0:
            pos_0 = np.array(es.card_positions[choices[0]])
            _draw_choice_border(frame, int(pos_0[0]), int(pos_0[1]),
                                agent0_color, border_thickness)
        if choices[1] >= 0:
            pos_1 = np.array(es.card_positions[choices[1]])
            _draw_choice_border(frame, int(pos_1[0]), int(pos_1[1]),
                                agent1_color, border_thickness)

        frames.append(frame)
    return frames


def _draw_choice_border(img, row, col, color, thickness):
    """Draw a thick border around a tile at grid (row, col) on an upscaled image."""
    tile_h = img.shape[0] // GRID_ROWS
    tile_w = img.shape[1] // GRID_COLS

    y0 = row * tile_h
    x0 = col * tile_w

    img[y0:y0 + thickness, x0:x0 + tile_w] = color
    img[y0 + tile_h - thickness:y0 + tile_h, x0:x0 + tile_w] = color
    img[y0:y0 + tile_h, x0:x0 + thickness] = color
    img[y0:y0 + tile_h, x0 + tile_w - thickness:x0 + tile_w] = color
