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


def test_ego_highlight():
    """Each agent's observation should differ (different ego borders)."""
    env = make_env("card-game", {})
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    assert not jnp.array_equal(obs["agent_0"], obs["agent_1"])


def test_print_observations():
    """Print rendered observations for visual inspection (run with pytest -s)."""
    import numpy as np

    env = make_env("card-game", {})
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    h = GRID_ROWS * TILE_PIXELS
    w = GRID_COLS * TILE_PIXELS

    perm = state.env_state.card_permutation
    card_names = ["RED", "BLUE", "GREEN", "YELLOW", "PURPLE"]
    print(f"\nCard permutation: {[card_names[int(p)] for p in perm]}")

    for agent in ["agent_0", "agent_1"]:
        img = (np.array(obs[agent]) * 255).astype(np.uint8).reshape(h, w, 3)
        print(f"\n{'='*60}")
        print(f"{agent} observation ({h}×{w} RGB)")
        print(f"{'='*60}")

        # Print as ASCII art: one char per pixel, using color approximation
        for row in range(h):
            line = []
            for col in range(w):
                r, g, b = img[row, col]
                if r == 0 and g == 0 and b == 0:
                    line.append(".")  # black / empty
                elif r == 255 and g == 0 and b == 255:
                    line.append("M")  # magenta ego border
                elif r > 180 and g < 100 and b < 100:
                    line.append("R")  # red-ish (agent 0 or card)
                elif r < 100 and g < 150 and b > 180:
                    line.append("B")  # blue-ish (agent 1 or card)
                elif r < 100 and g > 150 and b < 100:
                    line.append("G")  # green card
                elif r > 180 and g > 150 and b < 100:
                    line.append("Y")  # yellow card
                elif r > 120 and g < 100 and b > 150:
                    line.append("P")  # purple card
                elif r > 100 and g > 100 and b > 100:
                    line.append("#")  # other bright
                else:
                    line.append("?")  # unknown
            print("".join(line))

        print(f"\nLegend: . = black, M = magenta border, R = red, B = blue, "
              f"G = green, Y = yellow, P = purple")
