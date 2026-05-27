"""JAX renderer for the Hanabi environment.

Produces per-agent egocentric image observations from a JaxMARL `HanabiState`.
The image is laid out as public-state bands:
  - row 0: partner hand (5 cells, full identity visible)
  - row 1: fireworks (5 stacks, one per colour, showing the top placed rank)
  - row 2: info + life token strip (thermometer-style dots)
  - row 3: own hand (5 cells; identity hidden unless hints reveal it)
  - row 4: last action summary
  - rows 5+: deck remaining and discard count grid
  - bottom rows: exact JaxMARL symbolic observation rasterized as bits

Each card cell is 14 wide x 7 tall. Within a cell a 12x5 coloured rectangle
sits at the top, a 1-pixel gap follows, and a 1-pixel hint stripe at the
bottom encodes the colour-hint state (left half) and rank-hint state
(right half) the holder of that card has received.

The renderer is pure JAX (no `print`, no host-side branches on traced
values) and returns `uint8` `(H, W, 3)`. All colour pixels for cards,
fireworks and colour-hint stripes are drawn from `HANABI_COLORS`. Token
dots and card-backs use auxiliary colours that are *not* in
`HANABI_COLORS` so the Other-Play pixel-match recolouring wrapper can
permute the card palette without touching them.
"""
from functools import partial

import jax
import jax.numpy as jnp


TILE_PIXELS = 7

GRID_COLS = 10

HAND_SIZE = 5
NUM_COLORS = 5
NUM_RANKS = 5
MAX_INFO_TOKENS = 8
MAX_LIFE_TOKENS = 3
DECK_MAX = 40
SYMBOLIC_OBS_SIZE = 658

IMG_W = GRID_COLS * TILE_PIXELS  # 70
CELL_W = 14  # 2 cols
CELL_H = 7   # 1 row
ROW_PARTNER = 0
ROW_FIREWORKS = 1
ROW_TOKENS = 2
ROW_OWN = 3
ROW_LAST_ACTION = 4
DECK_Y0 = 35
DECK_H = 4
DISCARD_Y0 = 39
DISCARD_CELL_W = IMG_W // NUM_COLORS  # 14
VISUAL_IMG_H = DISCARD_Y0 + NUM_RANKS * CELL_H  # 74
SYMBOLIC_Y0 = VISUAL_IMG_H
SYMBOLIC_ROWS = (SYMBOLIC_OBS_SIZE + IMG_W - 1) // IMG_W
IMG_H = VISUAL_IMG_H + SYMBOLIC_ROWS  # 84

# Per-cell internal geometry
CARD_RECT_X0 = 1
CARD_RECT_Y0 = 0
CARD_RECT_W = 12
CARD_RECT_H = 5
HINT_STRIPE_Y = 6
HINT_STRIPE_X0 = 1
HINT_STRIPE_W = 12   # split into two 6-px halves: colour | rank


# 5 maximally distinct card colours, indexed to match JaxMARL's
# default colour_map = ["R", "Y", "G", "W", "B"].
HANABI_COLORS = jnp.array([
    [220, 50, 50],     # 0: R - red
    [220, 200, 50],    # 1: Y - yellow
    [50, 180, 50],     # 2: G - green
    [230, 230, 230],   # 3: W - white (rendered light gray for visibility)
    [50, 100, 220],    # 4: B - blue
], dtype=jnp.uint8)

# Auxiliary colours, none of which may coincide with a HANABI_COLORS entry —
# otherwise the Other-Play recolouring wrapper would silently permute them.
CARD_BACK_COLOR = jnp.array([90, 90, 90], dtype=jnp.uint8)
INFO_TOKEN_COLOR = jnp.array([170, 170, 170], dtype=jnp.uint8)
LIFE_TOKEN_COLOR = jnp.array([180, 0, 0], dtype=jnp.uint8)
RANK_HINT_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)  # rank-known indicator
DECK_BAR_COLOR = jnp.array([50, 180, 180], dtype=jnp.uint8)
ACTION_DISCARD_COLOR = jnp.array([180, 90, 0], dtype=jnp.uint8)
ACTION_PLAY_COLOR = jnp.array([120, 60, 200], dtype=jnp.uint8)
ACTION_HINT_COLOR = jnp.array([0, 160, 160], dtype=jnp.uint8)
ACTION_NOOP_COLOR = jnp.array([60, 60, 60], dtype=jnp.uint8)
ACTION_MARK_COLOR = jnp.array([255, 140, 0], dtype=jnp.uint8)
ACTION_SCORE_COLOR = jnp.array([0, 220, 120], dtype=jnp.uint8)
BACKGROUND_COLOR = jnp.array([0, 0, 0], dtype=jnp.uint8)


# 3x5 LCD-style digit glyphs (digits 0-9). For Hanabi we only use 1..5.
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
DIGIT_X_IN_CELL = 5   # centre the digit horizontally inside the 12-wide rect
DIGIT_Y_IN_CELL = 0


def _row_y(row_idx: int) -> int:
    """Top-y pixel of `row_idx` in the global image."""
    return row_idx * TILE_PIXELS


def _cell_x(slot_idx: int) -> int:
    """Top-x pixel of card-slot `slot_idx` in the global image."""
    return slot_idx * CELL_W


def _paint_rect(img, y0, x0, h, w, color):
    """Fill an (h, w) rectangle of `img` at (y0, x0) with `color`."""
    patch = jnp.broadcast_to(color, (h, w, 3))
    return jax.lax.dynamic_update_slice(img, patch, (y0, x0, 0))


def _stamp_digit(img, y0, x0, digit, color, on_when=True):
    """Stamp a single 3x5 digit glyph onto `img` at (y0, x0).

    `digit` is an int in [0, 9] (a traced jnp scalar is fine — we use
    take). `on_when` (a bool/traced scalar) gates whether the stamp is
    applied; when False, the image is returned unchanged.
    """
    glyph = PIXEL_DIGITS[digit]  # (5, 3)
    mask = glyph[:, :, None].astype(jnp.bool_)
    region = jax.lax.dynamic_slice(img, (y0, x0, 0), (DIGIT_H, DIGIT_W, 3))
    coloured = jnp.where(mask, color[None, None, :], region)
    new_region = jnp.where(on_when, coloured, region)
    return jax.lax.dynamic_update_slice(img, new_region, (y0, x0, 0))


def _decode_card(card_onehot):
    """Decode a (num_colors, num_ranks) one-hot card matrix.

    Returns (colour_idx, rank_idx, present) where `present` is True iff the
    matrix has any 1 (zero-filled padding cards decode to colour=0, rank=0,
    present=False).
    """
    flat = card_onehot.reshape(-1)
    present = flat.sum() > 0
    idx = jnp.argmax(flat)
    colour = idx // NUM_RANKS
    rank = idx % NUM_RANKS
    return colour, rank, present


def _fireworks_top_rank(fireworks_row):
    """Return (top_rank_idx, any_played) from a (num_ranks,) thermometer row.

    `fireworks[c]` is a thermometer: positions [0..top] are 1, rest 0. The
    top rank is the largest index with a 1.
    """
    any_played = jnp.any(fireworks_row)
    # argmax of reversed array gives the offset from the end of the highest set bit
    rev = jnp.flip(fireworks_row)
    top_offset = jnp.argmax(rev)
    top_idx = NUM_RANKS - 1 - top_offset
    return top_idx, any_played


def _render_card_cell(img, row_idx, slot_idx, fill_color, rank_idx, show_rank,
                      colour_hint, has_colour_hint, has_rank_hint):
    """Render one card cell (14x7) into `img` at (row_idx, slot_idx).

    Layout within the cell:
      - 12x5 fill_color rectangle at (y=0, x=1)
      - rank digit (3x5) stamped on top of the fill if show_rank
      - bottom-row hint stripe at y=6:
          left 6 px:  HANABI_COLORS[colour_hint] when has_colour_hint else background
          right 6 px: RANK_HINT_COLOR when has_rank_hint else background

    The hint stripe encodes what the *holder* of this card has been told,
    not what the viewer knows.
    """
    y0 = _row_y(row_idx)
    x0 = _cell_x(slot_idx)

    # card body
    img = _paint_rect(img, y0 + CARD_RECT_Y0, x0 + CARD_RECT_X0,
                      CARD_RECT_H, CARD_RECT_W, fill_color)

    # rank digit (centered in the 12-wide rect).
    # rank_idx is 0-indexed (0..NUM_RANKS-1); Hanabi displays ranks 1..NUM_RANKS,
    # so stamp the glyph for rank_idx + 1.
    img = _stamp_digit(
        img,
        y0 + CARD_RECT_Y0 + DIGIT_Y_IN_CELL,
        x0 + DIGIT_X_IN_CELL,
        rank_idx + 1,
        BACKGROUND_COLOR,  # punch the digit out of the coloured fill
        on_when=show_rank,
    )

    # hint stripe: left half = colour hint, right half = rank hint
    stripe_y = y0 + HINT_STRIPE_Y
    stripe_x = x0 + HINT_STRIPE_X0
    colour_pixel = HANABI_COLORS[colour_hint]
    img = jax.lax.cond(
        has_colour_hint,
        lambda im: _paint_rect(im, stripe_y, stripe_x, 1, HINT_STRIPE_W // 2, colour_pixel),
        lambda im: im,
        img,
    )
    img = jax.lax.cond(
        has_rank_hint,
        lambda im: _paint_rect(im, stripe_y, stripe_x + HINT_STRIPE_W // 2,
                               1, HINT_STRIPE_W // 2, RANK_HINT_COLOR),
        lambda im: im,
        img,
    )
    return img


def _render_partner_row(img, state, partner_idx):
    """Render partner_idx's hand into row 0 — full identity visible."""
    for slot in range(HAND_SIZE):
        card = state.player_hands[partner_idx, slot]
        colour, rank, present = _decode_card(card)
        cols_rev = state.colors_revealed[partner_idx, slot]
        ranks_rev = state.ranks_revealed[partner_idx, slot]
        has_col_hint = jnp.any(cols_rev)
        has_rank_hint = jnp.any(ranks_rev)
        col_hint_idx = jnp.argmax(cols_rev)
        # absent (padding) cards render as a black cell — fill becomes background
        fill = jnp.where(present, HANABI_COLORS[colour], BACKGROUND_COLOR)
        img = _render_card_cell(
            img, row_idx=ROW_PARTNER, slot_idx=slot,
            fill_color=fill, rank_idx=rank, show_rank=present,
            colour_hint=col_hint_idx, has_colour_hint=has_col_hint,
            has_rank_hint=has_rank_hint,
        )
    return img


def _render_fireworks_row(img, state):
    """Render the fireworks row — one cell per colour, showing top placed rank."""
    for colour in range(NUM_COLORS):
        top_rank, any_played = _fireworks_top_rank(state.fireworks[colour])
        fill = jnp.where(any_played, HANABI_COLORS[colour], BACKGROUND_COLOR)
        # no hint stripes on fireworks cells; pass dummy values gated off
        img = _render_card_cell(
            img, row_idx=ROW_FIREWORKS, slot_idx=colour,
            fill_color=fill, rank_idx=top_rank, show_rank=any_played,
            colour_hint=jnp.int32(0), has_colour_hint=jnp.bool_(False),
            has_rank_hint=jnp.bool_(False),
        )
    return img


def _render_token_row(img, state):
    """Render the token row — info tokens on the left, life tokens on the right.

    Each token is a 2-pixel-tall, 4-pixel-wide block with a 1-pixel gap.
    Info tokens span the left third of the row; life tokens the right third.
    Thermometer count = number of filled blocks.
    """
    row_y = _row_y(ROW_TOKENS)
    n_info = state.info_tokens.sum().astype(jnp.int32)
    n_life = state.life_tokens.sum().astype(jnp.int32)

    block_w = 4
    block_h = 3
    block_y = row_y + (CELL_H - block_h) // 2

    # info token strip: 8 blocks starting at x=2
    info_x0 = 2
    for i in range(MAX_INFO_TOKENS):
        x = info_x0 + i * (block_w + 1)
        on = jnp.int32(i) < n_info
        img = jax.lax.cond(
            on,
            lambda im, x=x: _paint_rect(im, block_y, x, block_h, block_w, INFO_TOKEN_COLOR),
            lambda im, x=x: im,
            img,
        )

    # life token strip: 3 blocks, right-aligned (last block ends at x=IMG_W-2)
    life_block_w = 4
    life_x_end = IMG_W - 2
    for i in range(MAX_LIFE_TOKENS):
        # rightmost (i=MAX-1) is drawn farthest right
        x = life_x_end - (MAX_LIFE_TOKENS - i) * (life_block_w + 1) + 1
        on = jnp.int32(i) < n_life
        img = jax.lax.cond(
            on,
            lambda im, x=x: _paint_rect(im, block_y, x, block_h, life_block_w, LIFE_TOKEN_COLOR),
            lambda im, x=x: im,
            img,
        )
    return img


def _render_own_row(img, state, agent_idx):
    """Render agent_idx's own hand into row 3 — identity hidden.

    Card-back colour by default. If a colour hint has been received, tint
    the card to that HANABI_COLORS entry (so the policy can attend to its
    own colour-known slots the same way it attends to partner-card colours).
    If a rank hint has been received, stamp the rank digit.
    """
    for slot in range(HAND_SIZE):
        cols_rev = state.colors_revealed[agent_idx, slot]
        ranks_rev = state.ranks_revealed[agent_idx, slot]
        has_col_hint = jnp.any(cols_rev)
        has_rank_hint = jnp.any(ranks_rev)
        col_hint_idx = jnp.argmax(cols_rev)
        rank_hint_idx = jnp.argmax(ranks_rev)

        # Also check the card is present (own padding cards exist when the deck runs out)
        card = state.player_hands[agent_idx, slot]
        present = card.sum() > 0

        # Fill: HANABI_COLORS[hinted_colour] if colour-hinted, else card-back
        fill = jnp.where(has_col_hint, HANABI_COLORS[col_hint_idx], CARD_BACK_COLOR)
        # absent cards render as background
        fill = jnp.where(present, fill, BACKGROUND_COLOR)

        img = _render_card_cell(
            img, row_idx=ROW_OWN, slot_idx=slot,
            fill_color=fill, rank_idx=rank_hint_idx,
            show_rank=present & has_rank_hint,
            colour_hint=col_hint_idx, has_colour_hint=has_col_hint,
            has_rank_hint=has_rank_hint,
        )
    return img


def _render_deck_bar(img, state):
    """Horizontal deck-remaining thermometer bar."""
    remaining = state.deck.sum().astype(jnp.int32)
    bar_w = (remaining * IMG_W) // DECK_MAX
    mask = jnp.arange(IMG_W) < bar_w
    full_bar = jnp.broadcast_to(DECK_BAR_COLOR, (DECK_H, IMG_W, 3))
    region = jax.lax.dynamic_slice(img, (DECK_Y0, 0, 0), (DECK_H, IMG_W, 3))
    new_region = jnp.where(mask[None, :, None], full_bar, region)
    return jax.lax.dynamic_update_slice(img, new_region, (DECK_Y0, 0, 0))


def _render_discard_grid(img, state):
    """Discard pile as a 5-rank by 5-colour count grid.

    Rows are ranks 1..5, columns are colours R/Y/G/W/B. A non-empty cell is
    filled with that card colour and stamped with the discarded count.
    """
    counts = state.discard_pile.sum(axis=0).astype(jnp.int32)  # (colour, rank)
    for rank in range(NUM_RANKS):
        y0 = DISCARD_Y0 + rank * CELL_H
        for colour in range(NUM_COLORS):
            x0 = colour * CELL_W
            count = counts[colour, rank]
            present = count > 0
            fill = jnp.where(present, HANABI_COLORS[colour], BACKGROUND_COLOR)
            img = _paint_rect(
                img,
                y0 + CARD_RECT_Y0,
                x0 + CARD_RECT_X0,
                CARD_RECT_H,
                CARD_RECT_W,
                fill,
            )
            img = _stamp_digit(
                img,
                y0 + CARD_RECT_Y0 + DIGIT_Y_IN_CELL,
                x0 + DIGIT_X_IN_CELL,
                count,
                BACKGROUND_COLOR,
                on_when=present,
            )
    return img


def _action_type_color(action):
    is_discard = (action >= 0) & (action < HAND_SIZE)
    is_play = (action >= HAND_SIZE) & (action < 2 * HAND_SIZE)
    is_hint = (action >= 2 * HAND_SIZE) & (action < 2 * HAND_SIZE + NUM_COLORS + NUM_RANKS)
    color = jnp.where(is_discard, ACTION_DISCARD_COLOR, ACTION_NOOP_COLOR)
    color = jnp.where(is_play, ACTION_PLAY_COLOR, color)
    return jnp.where(is_hint, ACTION_HINT_COLOR, color)


def _render_last_action_row(img, state, agent_idx, old_state, action, show_last_action):
    """Compact summary of JaxMARL's last-action features.

    Slot 0 marks actor-relative-to-viewer, slot 1 marks move type, slots 2-6
    mark affected/played hand positions, slot 7 shows hinted colour/rank or
    played/discarded card identity, and slots 8-9 mark play score / info gain.
    """
    row_y = _row_y(ROW_LAST_ACTION)
    original_region = jax.lax.dynamic_slice(img, (row_y, 0, 0), (CELL_H, IMG_W, 3))

    actor_idx = jnp.nonzero(old_state.cur_player_idx, size=1)[0][0]
    target_idx = 1 - actor_idx
    actor_is_self = actor_idx == agent_idx
    actor_fill = jnp.where(actor_is_self, INFO_TOKEN_COLOR, LIFE_TOKEN_COLOR)
    img = _render_card_cell(
        img, ROW_LAST_ACTION, 0, actor_fill, jnp.int32(0), jnp.bool_(False),
        jnp.int32(0), jnp.bool_(False), jnp.bool_(False),
    )

    action = jnp.asarray(action, dtype=jnp.int32)
    is_discard = (action >= 0) & (action < HAND_SIZE)
    is_play = (action >= HAND_SIZE) & (action < 2 * HAND_SIZE)
    is_hint_color = (action >= 2 * HAND_SIZE) & (action < 2 * HAND_SIZE + NUM_COLORS)
    is_hint_rank = (
        (action >= 2 * HAND_SIZE + NUM_COLORS)
        & (action < 2 * HAND_SIZE + NUM_COLORS + NUM_RANKS)
    )
    is_hint = is_hint_color | is_hint_rank

    img = _render_card_cell(
        img, ROW_LAST_ACTION, 1, _action_type_color(action), jnp.int32(0),
        jnp.bool_(False), jnp.int32(0), jnp.bool_(False), jnp.bool_(False),
    )

    pos = action % HAND_SIZE
    hint_idx = jnp.where(is_hint_color, action - 2 * HAND_SIZE, action - 2 * HAND_SIZE - NUM_COLORS)
    safe_hint_idx = jnp.clip(hint_idx, 0, NUM_COLORS - 1)
    target_hand = state.player_hands[target_idx]
    colour_reveal = target_hand.sum(axis=2)[:, safe_hint_idx]
    rank_reveal = target_hand.sum(axis=1)[:, safe_hint_idx]
    reveal_outcome = jnp.where(is_hint_color, colour_reveal, rank_reveal) > 0

    for slot in range(HAND_SIZE):
        affected = jnp.where(is_hint, reveal_outcome[slot], jnp.int32(slot) == pos)
        fill = jnp.where(affected, ACTION_MARK_COLOR, BACKGROUND_COLOR)
        img = _render_card_cell(
            img, ROW_LAST_ACTION, slot + 2, fill, jnp.int32(0), jnp.bool_(False),
            jnp.int32(0), jnp.bool_(False), jnp.bool_(False),
        )

    played_card = old_state.player_hands[actor_idx, pos]
    played_colour, played_rank, played_present = _decode_card(played_card)
    detail_colour = jnp.where(is_hint_color, HANABI_COLORS[safe_hint_idx], CARD_BACK_COLOR)
    detail_colour = jnp.where(is_play | is_discard, HANABI_COLORS[played_colour], detail_colour)
    detail_rank = jnp.where(is_hint_rank, safe_hint_idx, played_rank)
    show_detail_rank = is_hint_rank | ((is_play | is_discard) & played_present)
    img = _render_card_cell(
        img, ROW_LAST_ACTION, 7, detail_colour, detail_rank, show_detail_rank,
        jnp.int32(0), jnp.bool_(False), jnp.bool_(False),
    )

    card_played_score = state.fireworks.sum() > old_state.fireworks.sum()
    added_info_tokens = state.info_tokens.sum() > old_state.info_tokens.sum()
    score_fill = jnp.where(is_play & card_played_score, ACTION_SCORE_COLOR, BACKGROUND_COLOR)
    info_fill = jnp.where(is_play & added_info_tokens, INFO_TOKEN_COLOR, BACKGROUND_COLOR)
    img = _render_card_cell(
        img, ROW_LAST_ACTION, 8, score_fill, jnp.int32(0), jnp.bool_(False),
        jnp.int32(0), jnp.bool_(False), jnp.bool_(False),
    )
    img = _render_card_cell(
        img, ROW_LAST_ACTION, 9, info_fill, jnp.int32(0), jnp.bool_(False),
        jnp.int32(0), jnp.bool_(False), jnp.bool_(False),
    )

    rendered_region = jax.lax.dynamic_slice(img, (row_y, 0, 0), (CELL_H, IMG_W, 3))
    blended_region = jnp.where(show_last_action, rendered_region, original_region)
    return jax.lax.dynamic_update_slice(img, blended_region, (row_y, 0, 0))


def render_hanabi(state, agent_idx, old_state=None, action=None, show_last_action=False):
    """Render agent_idx's egocentric view of `state` as an (IMG_H, IMG_W, 3) uint8 image.

    For default 2-player Hanabi only — `partner_idx` is inferred as
    `1 - agent_idx`.
    """
    if old_state is None:
        old_state = state
    if action is None:
        action = jnp.int32(20)
    partner_idx = 1 - agent_idx
    img = jnp.broadcast_to(BACKGROUND_COLOR, (IMG_H, IMG_W, 3)).astype(jnp.uint8)
    img = _render_partner_row(img, state, partner_idx)
    img = _render_fireworks_row(img, state)
    img = _render_token_row(img, state)
    img = _render_own_row(img, state, agent_idx)
    img = _render_last_action_row(img, state, agent_idx, old_state, action, show_last_action)
    img = _render_deck_bar(img, state)
    img = _render_discard_grid(img, state)
    return img


def _unwrap_hanabi_state(state):
    """Descend through nested `env_state` fields until the inner HanabiState.

    The same renderer is used by the no-OP path (state is a `WrappedEnvState`)
    and the OP path (state is an `OPHanabiRecolouringState` wrapping the
    `WrappedEnvState`). Stops when there's no `env_state` attribute, which is
    the contract the inner JaxMARL `HanabiState` satisfies.
    """
    while hasattr(state, "env_state"):
        state = state.env_state
    return state


def render_hanabi_eval_frames(ep_states, scale: int = 8):
    """Render a list of wrapped episode states as side-by-side agent_0|agent_1 frames.

    Each frame is `(IMG_H*scale, 2*IMG_W*scale + sep, 3)` uint8. Used by the
    image_ippo / ja_ippo eval-video path; matches the shape contract that
    `render_card_game_eval_frames` provides for card-game.

    Renders in ground-truth colours, ignoring any per-agent OP recolouring;
    that's intentional — the OP recolouring is per-agent and the side-by-side
    view shows the public game state.
    """
    import numpy as np

    sep = 4
    frames = []
    for wrapped in ep_states:
        inner = _unwrap_hanabi_state(wrapped)
        img0 = np.asarray(render_hanabi(inner, jnp.int32(0)))
        img1 = np.asarray(render_hanabi(inner, jnp.int32(1)))
        # upscale by integer kron
        big0 = np.kron(img0, np.ones((scale, scale, 1), dtype=np.uint8))
        big1 = np.kron(img1, np.ones((scale, scale, 1), dtype=np.uint8))
        gap = np.full((big0.shape[0], sep, 3), 40, dtype=np.uint8)
        frames.append(np.concatenate([big0, gap, big1], axis=1))
    return frames


## Tests

def _smoke_render_to_png(out_path="/tmp/hanabi_render.png"):
    """Render a fresh-reset Hanabi state for agent 0 and write to PNG.

    Visual sanity check; not invoked under JIT. Run via:
        uv run python -c "from envs.hanabi.rendering import _smoke_render_to_png; _smoke_render_to_png()"
    """
    import numpy as np
    from PIL import Image
    from jaxmarl.environments.hanabi.hanabi import HanabiEnv

    env = HanabiEnv(
        num_agents=2, num_colors=5, num_ranks=5,
        max_info_tokens=8, max_life_tokens=3,
        num_cards_of_rank=np.array([3, 2, 2, 2, 1]),
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    img = render_hanabi(state, agent_idx=jnp.int32(0))
    arr = np.array(img)
    # upscale 8x for human eyeballing
    big = np.repeat(np.repeat(arr, 8, axis=0), 8, axis=1)
    Image.fromarray(big).save(out_path)
    print(f"wrote {out_path}  shape={arr.shape}")
