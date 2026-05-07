"""Visual + numerical check of message dots in the per-agent obs.

Runs a scripted card-game episode with `communication=True` and a deterministic
card permutation, then for each step saves:
  - obs_a0_step{t}.png — what agent_0 saw (decoded from the flat obs)
  - obs_a1_step{t}.png — what agent_1 saw
  - composite_step{t}.png — the side-by-side A0/A1 eval-video composite

Each agent's obs has a 2x2 white dot at the centre of the card the *partner*
last messaged about. The script asserts the dot lands on the column expected
from the partner's action and the current `card_permutation`. Saved PNGs are
useful when designing/debugging the message-visualisation pipeline.

Outputs land in `logs/message_obs_visual/`.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from envs import make_env
from envs.card_game.rendering import (
    CARD_RECT_H,
    CARD_RECT_W,
    CARD_RECT_Y,
    GRID_COLS,
    GRID_ROWS,
    NUM_CARDS,
    TILE_PIXELS,
    _unwrap_card_game_state,
    render_card_game_eval_frames,
)

_H = GRID_ROWS * TILE_PIXELS
_W = GRID_COLS * TILE_PIXELS
_UPSCALE = 32
_OUT_DIR = Path("logs/message_obs_visual")


def _flat_to_uint8_img(flat) -> np.ndarray:
    """(H*W*3,) float32 [0,1] -> (H, W, 3) uint8 RGB."""
    return np.asarray((np.asarray(flat).reshape(_H, _W, 3) * 255.0).astype(np.uint8))


def _upscale_nn(img: np.ndarray, factor: int = _UPSCALE) -> np.ndarray:
    return np.array(
        Image.fromarray(img).resize(
            (img.shape[1] * factor, img.shape[0] * factor), Image.NEAREST
        )
    )


def _find_white_dot_column(flat) -> int:
    """Return the card column (0-4) that contains a 2x2 all-white block in the
    card row of `flat`, or -1 if no such block exists.

    Restricts the search to the card row so the timestep digits at the top
    aren't picked up.
    """
    img = _flat_to_uint8_img(flat)
    card_band = img[
        int(CARD_RECT_Y) : int(CARD_RECT_Y) + CARD_RECT_H, :, :
    ]
    mask = (card_band[..., 0] == 255) & (card_band[..., 1] == 255) & (card_band[..., 2] == 255)

    for col in range(NUM_CARDS):
        x0 = 1 + col * (CARD_RECT_W + 2)
        x1 = x0 + CARD_RECT_W
        if mask[:, x0:x1].any():
            return col
    return -1


def _expected_partner_dot_col(partner_action: int, card_perm: np.ndarray) -> int:
    """Column where partner_action's *colour* lives in `card_perm`."""
    return int(np.where(card_perm == partner_action)[0][0])


def test_message_obs_visual():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in _OUT_DIR.glob("*.png"):
        f.unlink()

    env = make_env(
        "card-game",
        {"max_steps": 8, "communication": True, "shuffle": True},
    )
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    inner = _unwrap_card_game_state(state)
    card_perm = np.array(inner.card_permutation)
    print(f"card_permutation = {card_perm.tolist()}")

    # First obs (reset state) — neither agent has any partner-message yet.
    Image.fromarray(_upscale_nn(_flat_to_uint8_img(obs["agent_0"]))).save(
        _OUT_DIR / "obs_a0_step00.png"
    )
    Image.fromarray(_upscale_nn(_flat_to_uint8_img(obs["agent_1"]))).save(
        _OUT_DIR / "obs_a1_step00.png"
    )
    assert _find_white_dot_column(obs["agent_0"]) == -1
    assert _find_white_dot_column(obs["agent_1"]) == -1

    # Cycle through different colour pairs across the 7 deliberation steps and
    # take a final decision pick at step 8. The colour names below describe
    # which *colour identity* (0..4) each agent sends, not a position.
    deliberation_actions = [
        (0, 4),  # red, purple
        (2, 2),  # both green   -> match
        (3, 1),  # yellow, blue
        (4, 0),  # purple, red
        (1, 3),  # blue, yellow
        (2, 4),  # green, purple
        (0, 0),  # both red     -> match
    ]
    decision_action = (2, 2)  # both pick green at the decision step

    ep_states = [state]
    for t, (a0, a1) in enumerate(deliberation_actions, start=1):
        actions = {"agent_0": jnp.int32(a0), "agent_1": jnp.int32(a1)}
        key, sub = jax.random.split(key)
        obs, state, _, dones, _ = env.step(sub, state, actions)
        ep_states.append(state)
        assert not dones["__all__"], f"episode ended early at step {t}"

        # Each agent should now see the partner's message from THIS step.
        col_a0_seen = _find_white_dot_column(obs["agent_0"])
        col_a1_seen = _find_white_dot_column(obs["agent_1"])
        col_a0_expected = _expected_partner_dot_col(a1, card_perm)  # A0 sees A1's msg
        col_a1_expected = _expected_partner_dot_col(a0, card_perm)
        print(
            f"step {t:2d} | actions=(a0={a0}, a1={a1}) | "
            f"a0 dot col {col_a0_seen} (exp {col_a0_expected}) | "
            f"a1 dot col {col_a1_seen} (exp {col_a1_expected})"
        )
        assert col_a0_seen == col_a0_expected
        assert col_a1_seen == col_a1_expected

        Image.fromarray(_upscale_nn(_flat_to_uint8_img(obs["agent_0"]))).save(
            _OUT_DIR / f"obs_a0_step{t:02d}.png"
        )
        Image.fromarray(_upscale_nn(_flat_to_uint8_img(obs["agent_1"]))).save(
            _OUT_DIR / f"obs_a1_step{t:02d}.png"
        )

    # Decision step. The env auto-resets when `done`, so `state` after this
    # call is the *next* episode's reset, not the decision state. To
    # visualize the picks, we splice the picks into the last deliberation
    # state and use that as the final frame (still showing "08" since
    # step_count is preserved).
    actions = {"agent_0": jnp.int32(decision_action[0]),
               "agent_1": jnp.int32(decision_action[1])}
    key, sub = jax.random.split(key)
    _, _, _, dones, _ = env.step(sub, state, actions)
    assert dones["__all__"], "decision step should terminate the episode"
    last_delib = ep_states[-1]
    ep_states[-1] = last_delib.replace(
        env_state=last_delib.env_state.replace(
            agent_choices=jnp.array(decision_action, dtype=jnp.int32),
        )
    )

    # Composite eval frames across the full episode.
    frames = render_card_game_eval_frames(ep_states, scale=_UPSCALE)
    for t, frame in enumerate(frames):
        Image.fromarray(frame).save(_OUT_DIR / f"composite_step{t:02d}.png")

    print(f"\nSaved {2 * len(ep_states)} obs PNGs and {len(frames)} composite PNGs to {_OUT_DIR}")
