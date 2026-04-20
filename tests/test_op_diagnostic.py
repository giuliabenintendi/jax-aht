"""Diagnostic: verify OP recolouring + position shuffle on obs, messages, and picks.

Saves per-agent and GT views as PNGs, and checks the action-inversion pipeline
numerically for both deliberation (messages) and decision (picks) steps.

Run: uv run pytest -s tests/test_op_diagnostic.py
"""
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
    render_card_game,
)


_OUT_DIR = Path("tests/op_diagnostic")
_SCALE = 20


def _walk_to_base(state):
    """Walk .env_state chain to the inner CardGameState (has card_permutation)."""
    s = state
    while hasattr(s, "env_state"):
        s = s.env_state
    return s


def _save_flat_obs(flat_obs, path: Path) -> None:
    h = GRID_ROWS * TILE_PIXELS
    w = GRID_COLS * TILE_PIXELS
    img = (np.array(flat_obs) * 255).astype(np.uint8).reshape(h, w, 3)
    Image.fromarray(img).resize((w * _SCALE, h * _SCALE), Image.NEAREST).save(path)


def _save_rgb(img, path: Path) -> None:
    arr = np.array(img).astype(np.uint8)
    h, w = arr.shape[:2]
    Image.fromarray(arr).resize((w * _SCALE, h * _SCALE), Image.NEAREST).save(path)


def _find_dot_column(flat_obs, dot_rgb_u8) -> int | None:
    """Find the tile column in the card row showing the dot (exact pixel match)."""
    h = GRID_ROWS * TILE_PIXELS
    w = GRID_COLS * TILE_PIXELS
    img = (np.array(flat_obs) * 255).astype(np.uint8).reshape(h, w, 3)
    card_band = img[TILE_PIXELS:2 * TILE_PIXELS, :, :]
    mask = np.all(card_band == np.asarray(dot_rgb_u8, dtype=np.uint8), axis=-1)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    center_x = float(xs.mean())
    return int(center_x // TILE_PIXELS)


def _expected_dot_column(gt_msg: int, card_perm: np.ndarray, pos_perm: np.ndarray) -> int | None:
    """Column the dot should appear at in an agent's view for a given GT msg."""
    gt_positions = np.where(card_perm == gt_msg)[0]
    if len(gt_positions) == 0:
        return None
    gt_col = int(gt_positions[0])
    view_matches = np.where(pos_perm == gt_col)[0]
    if len(view_matches) == 0:
        return None
    return int(view_matches[0])


def test_op_diagnostic():
    _OUT_DIR.mkdir(exist_ok=True)

    env = make_env("card-game", {
        "max_steps": 8,
        "obs_type": "image",
        "shuffle": True,
        "communication": True,
        "other_play_position_shuffle": True,
        "other_play_recolouring": True,
    })

    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    # --- Reset-time per-agent vs GT views ---
    base = _walk_to_base(state)
    gt_img = render_card_game(base.card_permutation)

    _save_flat_obs(obs["agent_0"], _OUT_DIR / "reset_agent_0.png")
    _save_flat_obs(obs["agent_1"], _OUT_DIR / "reset_agent_1.png")
    _save_rgb(gt_img, _OUT_DIR / "reset_gt.png")

    recol_0 = np.array(state.per_agent_recolouring["agent_0"])
    recol_1 = np.array(state.per_agent_recolouring["agent_1"])
    inv_0 = np.array(state.per_agent_inv_recolouring["agent_0"])
    inv_1 = np.array(state.per_agent_inv_recolouring["agent_1"])
    pos_0 = np.array(state.env_state.per_agent_perm["agent_0"])
    pos_1 = np.array(state.env_state.per_agent_perm["agent_1"])
    card_perm = np.array(base.card_permutation)

    print("\n=== OP diagnostic (reset state) ===")
    print(f"  card_permutation (GT): {card_perm}   # column → GT colour")
    print(f"  agent_0 recolouring:   {recol_0}   inv: {inv_0}")
    print(f"  agent_1 recolouring:   {recol_1}   inv: {inv_1}")
    print(f"  agent_0 position perm: {pos_0}")
    print(f"  agent_1 position perm: {pos_1}")

    # --- Deliberation step: send known messages ---
    # Agent 0 picks agent-frame colour 2; agent 1 picks agent-frame colour 3.
    msg_frame_0 = 2
    msg_frame_1 = 3
    act = {
        "agent_0": jnp.int32(COMM_MESSAGE_BASE + msg_frame_0),
        "agent_1": jnp.int32(COMM_MESSAGE_BASE + msg_frame_1),
    }
    key, step_key = jax.random.split(key)
    obs, state, reward, done, info = env.step(step_key, state, act)

    gt_msg_0 = int(inv_0[msg_frame_0])
    gt_msg_1 = int(inv_1[msg_frame_1])
    base = _walk_to_base(state)
    stored = np.array(base.messages)

    print("\n=== Deliberation step 1 ===")
    print(f"  agent_0 sent frame {msg_frame_0} → expected GT {gt_msg_0}")
    print(f"  agent_1 sent frame {msg_frame_1} → expected GT {gt_msg_1}")
    print(f"  env-stored messages (GT):       {stored.tolist()}")

    assert int(stored[0]) == gt_msg_0, (
        f"agent_0 msg inversion wrong: stored {int(stored[0])}, expected {gt_msg_0}"
    )
    assert int(stored[1]) == gt_msg_1, (
        f"agent_1 msg inversion wrong: stored {int(stored[1])}, expected {gt_msg_1}"
    )

    _save_flat_obs(obs["agent_0"], _OUT_DIR / "delib_agent_0.png")
    _save_flat_obs(obs["agent_1"], _OUT_DIR / "delib_agent_1.png")
    _save_rgb(render_card_game(base.card_permutation), _OUT_DIR / "delib_gt.png")

    # --- Verify partner-message dot position in each agent's obs ---
    # Agent 0's obs should show agent 1's GT message as a magenta dot; the
    # dot column must match GT position of that colour mapped through agent
    # 0's position shuffle. Symmetric check for agent 1.
    card_perm_now = np.array(base.card_permutation)
    found_col_0 = _find_dot_column(obs["agent_0"], np.array(AGENT_1_COLOR))
    found_col_1 = _find_dot_column(obs["agent_1"], np.array(AGENT_0_COLOR))
    exp_col_0 = _expected_dot_column(gt_msg_1, card_perm_now, pos_0)
    exp_col_1 = _expected_dot_column(gt_msg_0, card_perm_now, pos_1)

    print("\n=== Dot-position check (post-deliberation) ===")
    print(f"  agent_0 view: dot expected at col {exp_col_0}, found at col {found_col_0}")
    print(f"  agent_1 view: dot expected at col {exp_col_1}, found at col {found_col_1}")

    assert found_col_0 == exp_col_0, (
        f"agent_0 dot position mismatch: expected {exp_col_0}, found {found_col_0}"
    )
    assert found_col_1 == exp_col_1, (
        f"agent_1 dot position mismatch: expected {exp_col_1}, found {found_col_1}"
    )

    # --- Fast-forward to decision step with noop messages ---
    for _ in range(6):
        key, step_key = jax.random.split(key)
        hold = {
            "agent_0": jnp.int32(COMM_MESSAGE_BASE + msg_frame_0),
            "agent_1": jnp.int32(COMM_MESSAGE_BASE + msg_frame_1),
        }
        obs, state, _, done, _ = env.step(step_key, state, hold)

    # --- Decision step: each agent picks their own frame colour 2 / 3 ---
    pick_frame_0 = 2
    pick_frame_1 = 3
    inv_0 = np.array(state.per_agent_inv_recolouring["agent_0"])
    inv_1 = np.array(state.per_agent_inv_recolouring["agent_1"])
    gt_pick_0 = int(inv_0[pick_frame_0])
    gt_pick_1 = int(inv_1[pick_frame_1])

    act = {
        "agent_0": jnp.int32(pick_frame_0),
        "agent_1": jnp.int32(pick_frame_1),
    }
    key, step_key = jax.random.split(key)
    obs, state, reward, done, info = env.step(step_key, state, act)

    base_post = _walk_to_base(state)
    # After episode end the env auto-resets; base now has reset state.
    # Read picks from the info that the step produced: env_raw reward fires
    # only when GT picks match. We can verify via the reward and manual check.
    r = float(reward["agent_0"])

    print("\n=== Decision step ===")
    print(f"  agent_0 picked frame {pick_frame_0} → expected GT {gt_pick_0}")
    print(f"  agent_1 picked frame {pick_frame_1} → expected GT {gt_pick_1}")
    print(f"  env reward (1 iff GT picks match):   {r}")
    print(f"  episode done:                          {bool(done['__all__'])}")

    expected_match = gt_pick_0 == gt_pick_1
    print(f"  expected match (GT picks equal):      {expected_match}")
    assert (r == 1.0) == expected_match, (
        f"reward/pick mismatch: reward={r}, GT picks {gt_pick_0} vs {gt_pick_1}"
    )

    print(f"\nSaved diagnostic PNGs to: {_OUT_DIR.resolve()}")
