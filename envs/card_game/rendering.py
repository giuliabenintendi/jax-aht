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


def _stamp_label_np(img_np, patterns, y, x, color, spacing=1, scale=1):
    """Stamp a sequence of 3x5 pixel-art glyphs left-to-right onto a numpy
    image. When `scale` > 1, `(y, x)` are raw obs coordinates and each glyph
    pixel is rendered as a `scale x scale` block — useful for stamping the
    same labels onto an already-upscaled video frame."""
    import numpy as np

    color_np = np.asarray(color, dtype=np.uint8)
    cx = x
    for pat in patterns:
        pat_np = np.asarray(pat, dtype=bool)
        h_pat, w_pat = pat_np.shape
        if scale == 1:
            region = img_np[y:y + h_pat, cx:cx + w_pat]
            region[pat_np] = color_np
        else:
            big = np.kron(pat_np, np.ones((scale, scale), dtype=bool))
            y0 = y * scale
            x0 = cx * scale
            region = img_np[y0:y0 + h_pat * scale, x0:x0 + w_pat * scale]
            region[big] = color_np
        cx += w_pat + spacing
    return img_np


def _draw_card_border_upscaled(img, card_pos, color, thickness, scale, outset_raw=1):
    """Draw a thick border around the card's rectangle.

    The border occupies a `thickness`-px strip on each edge of the card, then
    extends a further `outset_raw` raw obs-pixels outward into the surrounding
    black gap. This lets us render a *visually* very thick border without
    overwriting the inside of the card (cards are only 5 raw px wide, so an
    inside-only border quickly fills the whole card).
    """
    H, W = img.shape[:2]
    x0_card = (1 + card_pos * (CARD_RECT_W + 2)) * scale
    y0_card = int(CARD_RECT_Y) * scale
    w = CARD_RECT_W * scale
    h = CARD_RECT_H * scale
    out = max(0, outset_raw) * scale
    x0 = max(0, x0_card - out)
    y0 = max(0, y0_card - out)
    x1 = min(W, x0_card + w + out)
    y1 = min(H, y0_card + h + out)
    img[y0:y0 + thickness, x0:x1] = color
    img[y1 - thickness:y1, x0:x1] = color
    img[y0:y1, x0:x0 + thickness] = color
    img[y0:y1, x1 - thickness:x1] = color
    return img


def _gt_pick_to_view_col(state, agent_idx, gt_pick):
    """Map a GT-frame pick (colour ID) to the agent's view column. Under OP
    position shuffle this consults `per_agent_perm`; without OP it falls
    back to `card_permutation` (so view = physical layout). Returns -1 when
    there's no valid pick."""
    import numpy as np

    if gt_pick is None or int(gt_pick) < 0:
        return -1
    name = f"agent_{agent_idx}"
    s = state
    while s is not None and not hasattr(s, "per_agent_perm"):
        s = getattr(s, "env_state", None)
    if s is not None:
        pos_perm = np.asarray(s.per_agent_perm[name])
        matches = np.where(pos_perm == int(gt_pick))[0]
        if len(matches):
            return int(matches[0])
    base = state
    while base is not None and not hasattr(base, "card_permutation"):
        base = getattr(base, "env_state", None)
    if base is None:
        return -1
    perm = np.asarray(base.card_permutation)
    matches = np.where(perm == int(gt_pick))[0]
    return int(matches[0]) if len(matches) else -1


def _recolour_message_dot(base, state, agent_idx):
    """Mutates `base` (raw 21x35 RGB uint8) in place: replaces the white
    partner-message dot with the partner's video colour (A0=orange, A1=magenta).
    Eval-only — observations themselves keep the white dot.
    """
    import numpy as np

    inner = _unwrap_card_game_state(state)
    partner_msg_gt = int(np.asarray(inner.messages)[1 - agent_idx])
    if partner_msg_gt < 0:
        return base
    view_col = _gt_pick_to_view_col(state, agent_idx, partner_msg_gt)
    if view_col < 0:
        return base
    partner_color = np.array(
        AGENT_0_COLOR if (1 - agent_idx) == 0 else AGENT_1_COLOR, dtype=np.uint8,
    )
    dot_size = 2
    card_x = 1 + view_col * (CARD_RECT_W + 2)
    dot_y = int(CARD_RECT_Y) + (CARD_RECT_H - dot_size) // 2
    dot_x = card_x + (CARD_RECT_W - dot_size) // 2
    base[dot_y:dot_y + dot_size, dot_x:dot_x + dot_size] = partner_color
    return base


def render_card_game_gt_frame(state, last_action, is_decision, scale):
    """Render the canonical (un-OP'd) card scene with both agents' true
    broadcast messages and final picks overlaid in GT colour-id space.

    Lets a viewer check whether the two agents' signals actually converge on
    the same physical card, vs. coordinating by chance. A0 = orange, A1 =
    magenta; message dots sit near the top (A0) and bottom (A1) of each card.
    On the decision step, picks are drawn as colour-coded borders (A0 inner,
    A1 outer). `last_action` is the GT `(pick_0, pick_1)` tuple, used only
    when `is_decision` is True.
    """
    import numpy as np
    from PIL import Image

    inner = _unwrap_card_game_state(state)
    perm_np = np.asarray(inner.card_permutation)
    img = np.array(
        render_card_game_minimal(inner.card_permutation, inner.step_count + 1)
    )

    a0_color = np.array(AGENT_0_COLOR, dtype=np.uint8)
    a1_color = np.array(AGENT_1_COLOR, dtype=np.uint8)

    msgs = np.asarray(inner.messages)
    dot_size = 2
    for ai, color in ((0, a0_color), (1, a1_color)):
        msg = int(msgs[ai])
        if msg < 0:
            continue
        pos = int(np.argmin(np.abs(perm_np - msg)))
        card_x = 1 + pos * (CARD_RECT_W + 2)
        dot_x = card_x + (CARD_RECT_W - dot_size) // 2
        # A0 dot near the top of the card, A1 near the bottom, so both stay
        # visible when the two agents message the same card.
        dot_y = int(CARD_RECT_Y) + (1 if ai == 0 else CARD_RECT_H - dot_size - 1)
        img[dot_y:dot_y + dot_size, dot_x:dot_x + dot_size] = color

    sub = np.array(
        Image.fromarray(img).resize(
            (img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST,
        )
    )

    if is_decision and last_action is not None:
        for ai, color, outset in ((0, a0_color, 1), (1, a1_color, 3)):
            pick = int(last_action[ai])
            if pick < 0:
                continue
            matches = np.where(perm_np == pick)[0]
            if not len(matches):
                continue
            sub = _draw_card_border_upscaled(
                sub, int(matches[0]), color, 2 * scale, scale,
                outset_raw=outset,
            )
    return sub


def _render_one_agent_frame_from_obs(agent_idx, flat_obs, pick_view_col, scale,
                                      border_thickness, state=None,
                                      border_color=None):
    """Build a per-agent eval subview from the actual flat obs the policy saw.
    Under OP this naturally shows the agent's shuffled+recoloured view (the
    obs already contains the agent's message dot). Adds a colour-coded
    A0/A1 label and, when `pick_view_col >= 0`, a thick border at that
    view column — yellow if `border_color` is yellow (i.e. coordination
    success), white otherwise. When `state` is passed, the white message
    dot in the obs is recoloured to the partner's video colour."""
    import numpy as np
    from PIL import Image

    h_px = GRID_ROWS * TILE_PIXELS
    w_px = GRID_COLS * TILE_PIXELS
    base = (np.asarray(flat_obs).reshape(h_px, w_px, 3) * 255.0).astype(np.uint8)

    if state is not None:
        base = _recolour_message_dot(base, state, agent_idx)

    own_color = np.array(
        AGENT_0_COLOR if agent_idx == 0 else AGENT_1_COLOR, dtype=np.uint8,
    )
    label_pattern = _A0_PATTERN_SMALL if agent_idx == 0 else _A1_PATTERN_SMALL
    base = _stamp_label_np(base, label_pattern, 1, 27, own_color)

    sub = np.array(
        Image.fromarray(base).resize(
            (w_px * scale, h_px * scale), Image.NEAREST,
        )
    )

    if pick_view_col >= 0:
        if border_color is None:
            border_color = np.array([255, 255, 255], dtype=np.uint8)
        sub = _draw_card_border_upscaled(
            sub, pick_view_col, np.asarray(border_color, dtype=np.uint8),
            border_thickness, scale,
        )
    return sub


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

    choices = np.array(inner.agent_choices)
    own_pick = int(choices[agent_idx])
    other_pick = int(choices[1 - agent_idx])
    if own_pick >= 0:
        pos = int(np.where(perm_np == own_pick)[0][0])
        won = other_pick >= 0 and own_pick == other_pick
        border = np.array(
            [255, 255, 0] if won else [255, 255, 255], dtype=np.uint8,
        )
        sub = _draw_card_border_upscaled(
            sub, pos, border, border_thickness, scale,
        )
    return sub


def render_card_game_eval_frames_per_agent(ep_states, agent_idx: int, scale: int = 32,
                                            ep_obs=None, ep_actions=None):
    """Per-agent (single-game-width) eval frames. Used as the backdrop for
    attention-overlay grids. When `ep_obs` is provided, the base is the
    agent's actual obs (so under OP it shows their shuffled+recoloured view
    + their message dot)."""
    import numpy as np

    if agent_idx not in (0, 1):
        raise ValueError(f"agent_idx must be 0 or 1, got {agent_idx}")
    white = np.array([255, 255, 255], dtype=np.uint8)
    border_thickness = 3 * scale
    n = len(ep_states)
    out = []
    yellow = np.array([255, 255, 0], dtype=np.uint8)
    white_b = np.array([255, 255, 255], dtype=np.uint8)
    for t, state in enumerate(ep_states):
        if ep_obs is not None and t < len(ep_obs):
            view_col = -1
            border_color = white_b
            if t == n - 1 and ep_actions:
                state_for_perm = ep_states[t - 1] if t > 0 else state
                pick_gt = int(ep_actions[-1][agent_idx])
                view_col = _gt_pick_to_view_col(state_for_perm, agent_idx, pick_gt)
                pick_gt_0 = int(ep_actions[-1][0])
                pick_gt_1 = int(ep_actions[-1][1])
                won = pick_gt_0 >= 0 and pick_gt_1 >= 0 and pick_gt_0 == pick_gt_1
                border_color = yellow if won else white_b
            sub = _render_one_agent_frame_from_obs(
                agent_idx, ep_obs[t][f"agent_{agent_idx}"],
                view_col, scale, border_thickness,
                state=state, border_color=border_color,
            )
        else:
            sub = _render_one_agent_frame(
                _unwrap_card_game_state(state), agent_idx,
                scale, border_thickness, white,
            )
        out.append(sub)
    return out


def render_card_game_eval_frames(ep_states, scale: int = 32, gap_raw_px: int = 1,
                                  ep_obs=None, ep_actions=None):
    """Render stacked composite eval frames: A0's view on top, A1's view below.

    When `ep_obs` is provided, each subview uses the agent's *actual* flat
    obs as the base — under OP this reflects each agent's shuffled and
    recoloured view (the obs already contains the per-agent message dot
    too). Without `ep_obs` we fall back to the canonical-scene render.

    Args:
        ep_states: list of WrappedEnvState (from run_episode_with_states).
            Also works when other-play wrappers add extra nesting.
        scale: upscale factor (nearest-neighbor) for video quality.
        gap_raw_px: height of the black separator between the two subviews,
            in *raw* px (gets multiplied by scale).
        ep_obs: optional list of `{"agent_0": flat_obs, "agent_1": flat_obs}`
            dicts aligned with `ep_states` (one per timestep).
    """
    import numpy as np

    white = np.array([255, 255, 255], dtype=np.uint8)
    border_thickness = 2 * scale  # 2 raw px inside the card, plus 1 raw px outset
    gap_px = max(0, gap_raw_px) * scale

    frames = []
    n = len(ep_states)
    yellow = np.array([255, 255, 0], dtype=np.uint8)
    white_b = np.array([255, 255, 255], dtype=np.uint8)
    for t, state in enumerate(ep_states):
        inner = _unwrap_card_game_state(state)
        if ep_obs is not None and t < len(ep_obs):
            obs_t = ep_obs[t]
            view_col_0 = view_col_1 = -1
            border_color = white_b
            if t == n - 1 and ep_actions:
                state_for_perm = ep_states[t - 1] if t > 0 else state
                pick_gt_0 = int(ep_actions[-1][0])
                pick_gt_1 = int(ep_actions[-1][1])
                view_col_0 = _gt_pick_to_view_col(state_for_perm, 0, pick_gt_0)
                view_col_1 = _gt_pick_to_view_col(state_for_perm, 1, pick_gt_1)
                won = pick_gt_0 >= 0 and pick_gt_1 >= 0 and pick_gt_0 == pick_gt_1
                border_color = yellow if won else white_b
            sub_a0 = _render_one_agent_frame_from_obs(
                0, obs_t["agent_0"], view_col_0, scale, border_thickness,
                state=state, border_color=border_color,
            )
            sub_a1 = _render_one_agent_frame_from_obs(
                1, obs_t["agent_1"], view_col_1, scale, border_thickness,
                state=state, border_color=border_color,
            )
        else:
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
