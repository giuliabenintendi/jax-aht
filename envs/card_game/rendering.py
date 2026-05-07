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
WHITE_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)

_EMPTY_TILE = jnp.zeros((TILE_PIXELS, TILE_PIXELS, 3), dtype=jnp.uint8)

# 3x5 pixel digits 0-9 (LCD-style). Row-major: [digit, row, col], 1 = on.
PIXEL_DIGITS = jnp.array([
    [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],  # 0
    [[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]],  # 1
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],  # 2
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 3
    [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],  # 4
    [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 5
    [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # 6
    [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]],  # 7
    [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # 8
    [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 9
], dtype=jnp.uint8)
DIGIT_H, DIGIT_W = 5, 3

# Tall card rectangle: 5w x 7h, no rounding.
CARD_RECT_W, CARD_RECT_H = 5, 7
CARD_RECT_MASK = jnp.ones((CARD_RECT_H, CARD_RECT_W), dtype=jnp.bool_)

# Pixel-art letter "A", 3x5, matches PIXEL_DIGITS dimensions so labels like "A0"
# can be drawn the same way the timestep is drawn.
LETTER_A = jnp.array(
    [[0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=jnp.uint8
)

# Smaller 3x4 glyphs for the eval-video "A0" / "A1" inspection label so it sits
# more discreetly than the timestep counter.
LETTER_A_SMALL = jnp.array(
    [[0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=jnp.uint8
)
DIGIT_0_SMALL = jnp.array(
    [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.uint8
)
DIGIT_1_SMALL = jnp.array(
    [[0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=jnp.uint8
)

# Vertical position of cards in the minimal render (centered: (21-9)/2 = 6).
CARD_RECT_Y = (GRID_ROWS * TILE_PIXELS - CARD_RECT_H) // 2


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


def _stamp_digit(img, digit, y, x, color):
    """Stamp a single 3x5 digit on `img` at top-left (y, x). Only 'on' pixels are written."""
    pattern = PIXEL_DIGITS[digit]
    region = jax.lax.dynamic_slice(img, (y, x, 0), (DIGIT_H, DIGIT_W, 3))
    digit_img = jnp.where(pattern[:, :, None], color[None, None, :], region)
    return jax.lax.dynamic_update_slice(img, digit_img, (y, x, 0))


def _draw_timestep(img, step_count):
    """Draw a 2-digit step counter (00-99) flush to the top-left corner."""
    s = jnp.asarray(step_count, dtype=jnp.int32)
    digits = jnp.array([s // 10 % 10, s % 10])

    spacing = 1
    y_offset = 1
    x_offset = 1

    for i in range(2):
        img = _stamp_digit(
            img,
            digits[i],
            y_offset,
            x_offset + i * (DIGIT_W + spacing),
            WHITE_COLOR,
        )
    return img


def render_card_game_minimal(
    card_permutation: jnp.ndarray,
    step_count,
    revealed=None,
) -> jnp.ndarray:
    """Render the card game with cards only (no agent triangles) and a top-left
    pixel-art step counter.

    Layout (21x35 px):
      Top-left: 2-digit timestep counter (white, 3x4 LCD-style)
      Center:   5 vertical rounded-corner card rectangles (5w x 9h)
      Else:     black background
    """
    # Coerce so callers can pass plain numpy arrays — `jax.lax.scan` would
    # otherwise fail when indexing a numpy array with a traced loop var.
    card_permutation = jnp.asarray(card_permutation)
    if revealed is not None:
        revealed = jnp.asarray(revealed)

    h_px = GRID_ROWS * TILE_PIXELS
    w_px = GRID_COLS * TILE_PIXELS
    img = jnp.zeros((h_px, w_px, 3), dtype=jnp.uint8)

    empty_card = jnp.zeros((CARD_RECT_H, CARD_RECT_W, 3), dtype=jnp.uint8)
    mask = CARD_RECT_MASK[:, :, None]

    def draw_card(img, i):
        card_id = card_permutation[i]
        color = CARD_COLORS[card_id]
        if revealed is not None:
            color = jnp.where(revealed[i], color, GRAY_COLOR)
        card_tile = jnp.where(mask, color[None, None, :], empty_card)
        # Card x-origin: 1 px outer margin + i * (CARD_W + 2-px inner gap).
        x = 1 + i * (CARD_RECT_W + 2)
        img = jax.lax.dynamic_update_slice(img, card_tile, (CARD_RECT_Y, x, 0))
        return img, None

    img, _ = jax.lax.scan(draw_card, img, jnp.arange(NUM_CARDS))
    img = _draw_timestep(img, step_count)
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


def _unwrap_card_game_state(state):
    """Walk .env_state chain to find the base CardGameState."""
    s = state
    while hasattr(s, 'env_state') and not hasattr(s, 'card_permutation'):
        s = s.env_state
    return s


_A0_PATTERN = (LETTER_A, PIXEL_DIGITS[0])
_A1_PATTERN = (LETTER_A, PIXEL_DIGITS[1])
_A0_PATTERN_SMALL = (LETTER_A_SMALL, DIGIT_0_SMALL)
_A1_PATTERN_SMALL = (LETTER_A_SMALL, DIGIT_1_SMALL)


def _stamp_label_np(img_np, patterns, y, x, color, spacing=1):
    """Stamp a sequence of 3x5 pixel-art glyphs left-to-right onto a numpy image."""
    import numpy as np

    color_np = np.asarray(color, dtype=np.uint8)
    cx = x
    for pat in patterns:
        pat_np = np.asarray(pat, dtype=bool)
        h_pat, w_pat = pat_np.shape
        region = img_np[y:y + h_pat, cx:cx + w_pat]
        region[pat_np] = color_np
        cx += w_pat + spacing
    return img_np


def _draw_card_border_upscaled(img, card_pos, color, thickness, scale):
    """Draw a border that hugs the card's exact rectangle (no margin), drawn on
    top of the upscaled card pixels. Same physical size as the card itself."""
    x0 = (1 + card_pos * (CARD_RECT_W + 2)) * scale
    y0 = int(CARD_RECT_Y) * scale
    w = CARD_RECT_W * scale
    h = CARD_RECT_H * scale
    img[y0:y0 + thickness, x0:x0 + w] = color
    img[y0 + h - thickness:y0 + h, x0:x0 + w] = color
    img[y0:y0 + h, x0:x0 + thickness] = color
    img[y0:y0 + h, x0 + w - thickness:x0 + w] = color
    return img


def _render_one_agent_frame(inner, agent_idx, scale, border_thickness, _unused=None):
    """Render a single per-agent eval subview at the given upscale factor.

    Eval-only colour scheme (obs stays white): A0 = orange, A1 = magenta.
    The label, the partner-message dot (drawn in the *partner's* colour, so
    A0 sees A1's magenta dot and vice versa), and the own-pick border are
    all colour-coded so a viewer can tell who did what at a glance.
    """
    import numpy as np
    from PIL import Image

    own_color = np.array(
        AGENT_0_COLOR if agent_idx == 0 else AGENT_1_COLOR, dtype=np.uint8,
    )
    partner_color = np.array(
        AGENT_1_COLOR if agent_idx == 0 else AGENT_0_COLOR, dtype=np.uint8,
    )

    img = render_card_game_minimal(
        inner.card_permutation, inner.step_count + 1
    )
    img_np = np.array(img)

    perm_np = np.array(inner.card_permutation)
    partner_msg = int(inner.messages[1 - agent_idx])
    if partner_msg >= 0:
        msg_pos = int(np.argmin(np.abs(perm_np - partner_msg)))
        dot_size = 2
        card_x_origin = 1 + msg_pos * (CARD_RECT_W + 2)
        dot_y = int(CARD_RECT_Y) + (CARD_RECT_H - dot_size) // 2
        dot_x = card_x_origin + (CARD_RECT_W - dot_size) // 2
        img_np[dot_y:dot_y + dot_size, dot_x:dot_x + dot_size] = partner_color

    label_pattern = _A0_PATTERN_SMALL if agent_idx == 0 else _A1_PATTERN_SMALL
    img_np = _stamp_label_np(img_np, label_pattern, 1, 27, own_color)

    h, w = img_np.shape[:2]
    pil_img = Image.fromarray(img_np).resize(
        (w * scale, h * scale), Image.NEAREST
    )
    sub = np.array(pil_img)

    own_pick = int(np.array(inner.agent_choices)[agent_idx])
    if own_pick >= 0:
        pos = int(np.where(perm_np == own_pick)[0][0])
        sub = _draw_card_border_upscaled(
            sub, pos, own_color, border_thickness, scale
        )
    return sub


def render_card_game_eval_frames_per_agent(ep_states, agent_idx: int, scale: int = 32):
    """Per-agent (single-game-width) eval frames. Same content as one half of
    `render_card_game_eval_frames`'s composite. Useful as the backdrop for
    attention-overlay grids, which expect frames sized to the original obs."""
    import numpy as np

    if agent_idx not in (0, 1):
        raise ValueError(f"agent_idx must be 0 or 1, got {agent_idx}")
    white = np.array([255, 255, 255], dtype=np.uint8)
    border_thickness = scale  # 1 raw obs-pixel thick, matches obs-element scale
    return [
        _render_one_agent_frame(
            _unwrap_card_game_state(state), agent_idx, scale, border_thickness, white
        )
        for state in ep_states
    ]


def render_card_game_eval_frames(ep_states, scale: int = 32, gap_raw_px: int = 1):
    """Render stacked composite eval frames: A0's view on top, A1's view below.

    Each frame shows two mini card-games separated by a thin black gap. The
    top mini-game is what A0 saw (cards + timestep + A0's partner-message dot)
    with a thick white border around A0's chosen card and an "A0" label. The
    bottom is the symmetric thing for A1. Stacking both subviews in one frame
    makes "who picked what" readable without flipping between videos.

    Args:
        ep_states: list of WrappedEnvState (from run_episode_with_states).
            Also works when other-play wrappers add extra nesting.
        scale: upscale factor (nearest-neighbor) for video quality.
        gap_raw_px: height of the black separator between the two subviews,
            in *raw* px (gets multiplied by scale).

    Returns:
        list of (2*H_scaled + gap_raw_px*scale, W_scaled, 3) uint8 numpy arrays.
    """
    import numpy as np

    white = np.array([255, 255, 255], dtype=np.uint8)
    border_thickness = scale  # 1 raw obs-pixel thick, matches obs-element scale
    gap_px = max(0, gap_raw_px) * scale

    frames = []
    for state in ep_states:
        inner = _unwrap_card_game_state(state)
        sub_a0 = _render_one_agent_frame(inner, 0, scale, border_thickness, white)
        sub_a1 = _render_one_agent_frame(inner, 1, scale, border_thickness, white)

        if gap_px > 0:
            w_sub = sub_a0.shape[1]
            gap = np.zeros((gap_px, w_sub, 3), dtype=np.uint8)
            composite = np.concatenate([sub_a0, gap, sub_a1], axis=0)
        else:
            composite = np.concatenate([sub_a0, sub_a1], axis=0)
        frames.append(composite)

    return frames
