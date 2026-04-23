"""Visual verification of partner-message scrambling.

Runs two card-game envs with identical reset seed and identical actions,
one with `scramble_partner_msg=True` and one with it off, and renders
their obs side-by-side. Confirms numerically that the true-path dot lands
at the expected card column and that the scrambled path produces at least
one column change across the rollout. Saves the visual grids so the
intervention can be verified by eye.

Outputs:
  - logs/msg_scramble/step_NN.png — 2x2 grids
      row 0: agent_0 obs (true | scrambled)
      row 1: agent_1 obs (true | scrambled)
  - logs/msg_scramble/summary.txt — per-step dot-column table
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from envs import make_env
from envs.card_game.action_utils import COMM_MESSAGE_BASE
from envs.card_game.rendering import (
    AGENT_0_COLOR,
    AGENT_1_COLOR,
    GRID_COLS,
    GRID_ROWS,
    TILE_PIXELS,
)

_H = GRID_ROWS * TILE_PIXELS
_W = GRID_COLS * TILE_PIXELS
_UPSCALE = 16

_AGENT_0_RGB = tuple(int(c) for c in AGENT_0_COLOR)
_AGENT_1_RGB = tuple(int(c) for c in AGENT_1_COLOR)


def _flat_to_img(flat: jnp.ndarray) -> np.ndarray:
    return np.asarray((flat.reshape(_H, _W, 3) * 255.0).astype(jnp.uint8))


def _upscale(img: np.ndarray) -> np.ndarray:
    return np.kron(img, np.ones((_UPSCALE, _UPSCALE, 1), dtype=np.uint8))


def _dot_column(flat: jnp.ndarray, partner_rgb: tuple[int, int, int]) -> int:
    """Column (0-4) of the partner-color dot, or -1 if no dot was drawn.

    Restricts the search to the card row so it doesn't collide with the
    partner's own triangle pixels outside the card row.
    """
    img = _flat_to_img(flat)
    card_row = img[TILE_PIXELS : 2 * TILE_PIXELS]
    mask = (
        (card_row[..., 0] == partner_rgb[0])
        & (card_row[..., 1] == partner_rgb[1])
        & (card_row[..., 2] == partner_rgb[2])
    )
    cols = np.where(mask.any(axis=0))[0]
    if cols.size == 0:
        return -1
    return int(cols[0] // TILE_PIXELS)


def _compose_grid(
    a0_true: jnp.ndarray,
    a0_scr: jnp.ndarray,
    a1_true: jnp.ndarray,
    a1_scr: jnp.ndarray,
) -> np.ndarray:
    tiles = [
        _upscale(_flat_to_img(x)) for x in (a0_true, a0_scr, a1_true, a1_scr)
    ]
    h, w = tiles[0].shape[:2]
    sep = 4
    grid = np.full((2 * h + sep, 2 * w + sep, 3), 255, dtype=np.uint8)
    grid[:h, :w] = tiles[0]
    grid[:h, w + sep:] = tiles[1]
    grid[h + sep:, :w] = tiles[2]
    grid[h + sep:, w + sep:] = tiles[3]
    return grid


def test_scramble_partner_msg_visual() -> None:
    out_dir = Path("logs/msg_scramble")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_kwargs = {"communication": True, "max_steps": 6}
    env_true = make_env("card-game", dict(base_kwargs))
    env_scr = make_env(
        "card-game", dict(base_kwargs, scramble_partner_msg=True)
    )

    reset_key = jax.random.PRNGKey(0)
    _, st_true = env_true.reset(reset_key)
    _, st_scr = env_scr.reset(reset_key)
    perm = np.asarray(st_true.env_state.card_permutation)
    assert np.array_equal(perm, np.asarray(st_scr.env_state.card_permutation))

    msg_0, msg_1 = 2, 3
    actions = {
        "agent_0": jnp.int32(COMM_MESSAGE_BASE + msg_0),
        "agent_1": jnp.int32(COMM_MESSAGE_BASE + msg_1),
    }
    # agent 0 sees agent 1's msg (msg_1); agent 1 sees agent 0's msg (msg_0).
    expected_a0_col = int(np.where(perm == msg_1)[0][0])
    expected_a1_col = int(np.where(perm == msg_0)[0][0])

    lines = [
        f"card_permutation (pos -> color): {perm.tolist()}",
        f"agent_0 sends color {msg_0} -> true dot for agent_1 at col {expected_a1_col}",
        f"agent_1 sends color {msg_1} -> true dot for agent_0 at col {expected_a0_col}",
        "",
        "step | a0_true a0_scr | a1_true a1_scr",
    ]

    # With max_steps=6, the decision step is the 5th call. Keep strictly to
    # deliberation so message actions are legal and the env doesn't auto-reset.
    n_deliberation = env_true.max_steps - 2

    step_key = jax.random.PRNGKey(42)
    n_scr_changes = 0
    for t in range(n_deliberation):
        step_key, k_true, k_scr = jax.random.split(step_key, 3)
        obs_t, st_true, *_ = env_true.step(k_true, st_true, actions)
        obs_s, st_scr, *_ = env_scr.step(k_scr, st_scr, actions)

        col_true_a0 = _dot_column(obs_t["agent_0"], _AGENT_1_RGB)
        col_scr_a0 = _dot_column(obs_s["agent_0"], _AGENT_1_RGB)
        col_true_a1 = _dot_column(obs_t["agent_1"], _AGENT_0_RGB)
        col_scr_a1 = _dot_column(obs_s["agent_1"], _AGENT_0_RGB)

        assert col_true_a0 == expected_a0_col
        assert col_true_a1 == expected_a1_col

        n_scr_changes += int(col_scr_a0 != col_true_a0)
        n_scr_changes += int(col_scr_a1 != col_true_a1)

        grid = _compose_grid(
            obs_t["agent_0"], obs_s["agent_0"],
            obs_t["agent_1"], obs_s["agent_1"],
        )
        Image.fromarray(grid).save(out_dir / f"step_{t:02d}.png")

        lines.append(
            f"  {t:2d} |   {col_true_a0}       {col_scr_a0}    "
            f"|    {col_true_a1}       {col_scr_a1}"
        )

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    # 2*n_deliberation independent uniform draws from 5 colors. Probability
    # of zero column changes is (1/5)^(2*n_deliberation) — negligible.
    assert n_scr_changes > 0
