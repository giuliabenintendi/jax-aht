"""Basic tests for the Card Game environment."""
import jax
import jax.numpy as jnp

from envs import make_env
from envs.card_game.rendering import (
    render_card_game,
    TILE_PIXELS,
    GRID_ROWS,
    GRID_COLS,
    NUM_CARDS,
    CARD_COLORS,
)


def test_render_shape():
    perm = jnp.arange(NUM_CARDS)
    img = render_card_game(perm)
    assert img.shape == (GRID_ROWS * TILE_PIXELS, GRID_COLS * TILE_PIXELS, 3)
    assert img.dtype == jnp.uint8


def test_render_cards_differ():
    """Each card position should have distinct pixels (different colors)."""
    perm = jnp.arange(NUM_CARDS)
    img = render_card_game(perm)
    card_row = img[TILE_PIXELS : 2 * TILE_PIXELS]  # row 1
    tiles = [card_row[:, i * TILE_PIXELS : (i + 1) * TILE_PIXELS] for i in range(NUM_CARDS)]
    for i in range(NUM_CARDS):
        for j in range(i + 1, NUM_CARDS):
            assert not jnp.array_equal(tiles[i], tiles[j])


def test_render_shuffle_changes_layout():
    """Shuffled permutation should produce a different image."""
    perm_a = jnp.array([0, 1, 2, 3, 4])
    perm_b = jnp.array([4, 3, 2, 1, 0])
    img_a = render_card_game(perm_a)
    img_b = render_card_game(perm_b)
    assert not jnp.array_equal(img_a, img_b)


def test_make_env():
    env = make_env("card-game", {})
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    assert "agent_0" in obs
    assert "agent_1" in obs
    assert obs["agent_0"].shape == (GRID_ROWS * TILE_PIXELS * GRID_COLS * TILE_PIXELS * 3,)


def test_episode_length():
    """Episode should last exactly max_steps steps."""
    env = make_env("card-game", {"max_steps": 10})
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    for t in range(10):
        key, subkey = jax.random.split(key)
        actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
        obs, state, reward, dones, info = env.step(subkey, state, actions)

        if t < 9:
            assert not dones["__all__"], f"Episode ended early at step {t + 1}"
        else:
            assert dones["__all__"], "Episode should end at step 10"


def test_matching_reward():
    """Both agents picking the same position on decision step → reward 1."""
    env = make_env("card-game", {"max_steps": 10})
    key = jax.random.PRNGKey(7)
    obs, state = env.reset(key)

    # Run 9 deliberation steps
    for _ in range(9):
        key, subkey = jax.random.split(key)
        actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
        obs, state, reward, dones, info = env.step(subkey, state, actions)
        assert float(reward["agent_0"]) == 0.0

    # Decision step: both pick position 2
    key, subkey = jax.random.split(key)
    actions = {"agent_0": jnp.int32(2), "agent_1": jnp.int32(2)}
    obs, state, reward, dones, info = env.step(subkey, state, actions)
    assert float(reward["agent_0"]) == 1.0
    assert dones["__all__"]


def test_mismatching_reward():
    """Different positions on decision step → reward 0."""
    env = make_env("card-game", {"max_steps": 10})
    key = jax.random.PRNGKey(7)
    obs, state = env.reset(key)

    for _ in range(9):
        key, subkey = jax.random.split(key)
        actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
        obs, state, reward, dones, info = env.step(subkey, state, actions)

    key, subkey = jax.random.split(key)
    actions = {"agent_0": jnp.int32(1), "agent_1": jnp.int32(3)}
    obs, state, reward, dones, info = env.step(subkey, state, actions)
    assert float(reward["agent_0"]) == 0.0
    assert dones["__all__"]


def test_auto_reset():
    """After episode ends, state should auto-reset (new permutation possible)."""
    env = make_env("card-game", {"max_steps": 10})
    key = jax.random.PRNGKey(99)
    obs, state = env.reset(key)
    perm_before = state.env_state.card_permutation

    # Run full episode
    for _ in range(10):
        key, subkey = jax.random.split(key)
        actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
        obs, state, reward, dones, info = env.step(subkey, state, actions)

    # After auto-reset, step count should be 0
    assert int(state.env_state.step_count) == 0


def test_comm_action_space_and_masks():
    """Communication uses flat macro-actions with phase-specific masks.

    With max_steps=3, the decision step is step 3, so steps 1-2 are
    deliberation (picks masked, messages legal) and step 3 is decision
    (picks legal, messages masked).
    """
    env = make_env("card-game", {"max_steps": 3, "communication": True, "shuffle": False})
    assert env.action_space("agent_0").n == 2 * NUM_CARDS

    key = jax.random.PRNGKey(5)
    obs, state = env.reset(key)

    expected_deliberation = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=jnp.float32)
    expected_decision = jnp.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=jnp.float32)

    # Mask at step 1 (next_step = 1, still deliberation)
    assert jnp.array_equal(env.get_avail_actions(state)["agent_0"], expected_deliberation)

    # Take a deliberation step, mask at step 2 (next_step = 2, still deliberation)
    key, subkey = jax.random.split(key)
    msg_actions = {"agent_0": jnp.int32(NUM_CARDS), "agent_1": jnp.int32(NUM_CARDS + 1)}
    obs, state, _, dones, _ = env.step(subkey, state, msg_actions)
    assert not dones["__all__"]
    assert jnp.array_equal(env.get_avail_actions(state)["agent_0"], expected_deliberation)

    # Take another deliberation step, mask at step 3 (next_step = 3, decision)
    key, subkey = jax.random.split(key)
    obs, state, _, dones, _ = env.step(subkey, state, msg_actions)
    assert not dones["__all__"]
    assert jnp.array_equal(env.get_avail_actions(state)["agent_0"], expected_decision)


def test_comm_message_updates_state():
    """Message actions set the current message per agent."""
    env = make_env("card-game", {"max_steps": 3, "communication": True, "shuffle": False})
    key = jax.random.PRNGKey(11)
    obs, state = env.reset(key)

    key, subkey = jax.random.split(key)
    actions = {"agent_0": jnp.int32(NUM_CARDS + 2), "agent_1": jnp.int32(NUM_CARDS + 4)}
    obs, state, _, dones, _ = env.step(subkey, state, actions)
    assert not dones["__all__"]
    assert int(state.env_state.messages[0]) == 2
    assert int(state.env_state.messages[1]) == 4


def test_comm_decision_pick_reward():
    """Final-step picks are plain card actions under communication."""
    env = make_env("card-game", {"max_steps": 2, "communication": True, "shuffle": False})
    key = jax.random.PRNGKey(13)
    obs, state = env.reset(key)

    key, subkey = jax.random.split(key)
    msg = {"agent_0": jnp.int32(NUM_CARDS), "agent_1": jnp.int32(NUM_CARDS)}
    obs, state, _, dones, _ = env.step(subkey, state, msg)
    assert not dones["__all__"]

    key, subkey = jax.random.split(key)
    actions = {"agent_0": jnp.int32(3), "agent_1": jnp.int32(3)}
    obs, state, reward, dones, _ = env.step(subkey, state, actions)
    assert dones["__all__"]
    assert float(reward["agent_0"]) == 1.0


def test_ego_highlight():
    """Each agent's observation should differ (different ego borders)."""
    env = make_env("card-game", {})
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    assert not jnp.array_equal(obs["agent_0"], obs["agent_1"])


def test_save_observations():
    """Save rendered observations as upscaled PNGs for visual inspection."""
    import numpy as np
    from PIL import Image
    from pathlib import Path

    out_dir = Path("tests/card_game_obs")
    out_dir.mkdir(exist_ok=True)

    env = make_env("card-game", {})
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    h = GRID_ROWS * TILE_PIXELS
    w = GRID_COLS * TILE_PIXELS
    scale = 20  # upscale for visibility

    perm = state.env_state.card_permutation
    card_names = ["RED", "BLUE", "GREEN", "YELLOW", "PURPLE"]
    print(f"\nCard permutation: {[card_names[int(p)] for p in perm]}")

    for agent in ["agent_0", "agent_1"]:
        img = (np.array(obs[agent]) * 255).astype(np.uint8).reshape(h, w, 3)
        pil_img = Image.fromarray(img).resize(
            (w * scale, h * scale), Image.NEAREST
        )
        path = out_dir / f"{agent}.png"
        pil_img.save(path)
        print(f"Saved {path} ({w * scale}×{h * scale} px)")
