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


def test_match_reward_mirrors_partner_previous_message():
    """match shaping rewards agent i when its current message equals the
    partner's previous message — lagged and per-agent (see _comm_shaping)."""
    env = make_env("card-game", {"max_steps": 10, "communication": True,
                                 "match_coef": 0.1})
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Step 1: no previous partner message exists yet -> no match reward.
    key, subkey = jax.random.split(key)
    obs, state, _, _, info = env.step(
        subkey, state, {"agent_0": jnp.int32(2), "agent_1": jnp.int32(2)})
    assert float(info["comm_reward_match"][0]) == 0.0
    assert float(info["comm_reward_match"][1]) == 0.0

    # Step 2: each agent's current message equals the partner's previous (2).
    key, subkey = jax.random.split(key)
    obs, state, _, _, info = env.step(
        subkey, state, {"agent_0": jnp.int32(2), "agent_1": jnp.int32(2)})
    assert float(info["comm_reward_match"][0]) > 0.0
    assert float(info["comm_reward_match"][1]) > 0.0

    # Step 3: only agent 1 still sends the partner's previous message (2) ->
    # the reward is assigned per agent.
    key, subkey = jax.random.split(key)
    obs, state, _, _, info = env.step(
        subkey, state, {"agent_0": jnp.int32(4), "agent_1": jnp.int32(2)})
    assert float(info["comm_reward_match"][0]) == 0.0
    assert float(info["comm_reward_match"][1]) > 0.0


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


def test_comm_action_space_is_unified():
    """Unified intent-expression layout: Discrete(NUM_CARDS) at every step.

    All cards are always legal; the env's `is_decision` flag routes the
    same emitted card to either `messages` (deliberation) or `agent_choices`
    (decision).
    """
    env = make_env("card-game", {"max_steps": 3, "communication": True, "shuffle": False})
    assert env.action_space("agent_0").n == NUM_CARDS

    key = jax.random.PRNGKey(5)
    obs, state = env.reset(key)

    expected_mask = jnp.ones(NUM_CARDS, dtype=jnp.float32)
    assert jnp.array_equal(env.get_avail_actions(state)["agent_0"], expected_mask)

    key, subkey = jax.random.split(key)
    actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(1)}
    obs, state, _, dones, _ = env.step(subkey, state, actions)
    assert not dones["__all__"]
    assert jnp.array_equal(env.get_avail_actions(state)["agent_0"], expected_mask)


def test_comm_message_updates_state():
    """Deliberation-step actions populate the messages field."""
    env = make_env("card-game", {"max_steps": 3, "communication": True, "shuffle": False})
    key = jax.random.PRNGKey(11)
    obs, state = env.reset(key)

    key, subkey = jax.random.split(key)
    actions = {"agent_0": jnp.int32(2), "agent_1": jnp.int32(4)}
    obs, state, _, dones, _ = env.step(subkey, state, actions)
    assert not dones["__all__"]
    assert int(state.env_state.messages[0]) == 2
    assert int(state.env_state.messages[1]) == 4


def test_comm_decision_pick_reward():
    """Decision-step actions are picks under the unified layout."""
    env = make_env("card-game", {"max_steps": 2, "communication": True, "shuffle": False})
    key = jax.random.PRNGKey(13)
    obs, state = env.reset(key)

    key, subkey = jax.random.split(key)
    msg = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
    obs, state, _, dones, _ = env.step(subkey, state, msg)
    assert not dones["__all__"]

    key, subkey = jax.random.split(key)
    actions = {"agent_0": jnp.int32(3), "agent_1": jnp.int32(3)}
    obs, state, reward, dones, _ = env.step(subkey, state, actions)
    assert dones["__all__"]
    assert float(reward["agent_0"]) == 1.0


def test_following_partner_message_is_chance_without_follow_reward():
    """Following partner messages can yield 0.2 return without any shaping.

    Protocol:
    - step 1: each agent sends an arbitrary color message
    - step 2: each agent picks the partner's previous message

    With zero communication shaping, this still succeeds exactly when the two
    messages matched, i.e. 5 successes out of 25 message pairs.
    """
    env = make_env(
        "card-game",
        {
            "max_steps": 2,
            "communication": True,
            "shuffle": False,
            "match_coef": 0.0,
            "follow_coef": 0.0,
        },
    )

    reward_sum = 0.0
    num_pairs = 0
    key = jax.random.PRNGKey(123)

    for msg_0 in range(NUM_CARDS):
        for msg_1 in range(NUM_CARDS):
            key, reset_key, step1_key, step2_key = jax.random.split(key, 4)
            _, state = env.reset(reset_key)

            _, state, step1_reward, step1_dones, step1_info = env.step(
                step1_key,
                state,
                {
                    "agent_0": jnp.int32(msg_0),
                    "agent_1": jnp.int32(msg_1),
                },
            )
            assert not step1_dones["__all__"]
            assert float(step1_reward["agent_0"]) == 0.0
            assert float(step1_info["comm_reward_follow"][0]) == 0.0

            _, _, step2_reward, step2_dones, step2_info = env.step(
                step2_key,
                state,
                {
                    "agent_0": jnp.int32(msg_1),
                    "agent_1": jnp.int32(msg_0),
                },
            )
            assert step2_dones["__all__"]
            assert float(step2_info["comm_reward_follow"][0]) == 0.0
            assert float(step2_info["comm_reward"][0]) == 0.0

            reward_sum += float(step2_reward["agent_0"])
            num_pairs += 1

    assert num_pairs == NUM_CARDS * NUM_CARDS
    assert reward_sum == float(NUM_CARDS)
    assert reward_sum / num_pairs == 1.0 / NUM_CARDS


def test_communication_dot_makes_agent_obs_differ():
    """The only per-agent difference in the observation is the partner-message
    dot: with communication off both agents see the same frame; with it on,
    the frames diverge once the agents have sent distinct messages."""
    key = jax.random.PRNGKey(0)

    env_off = make_env("card-game", {"max_steps": 10, "shuffle": False})
    obs_off, _ = env_off.reset(key)
    assert jnp.array_equal(obs_off["agent_0"], obs_off["agent_1"])

    env_on = make_env(
        "card-game", {"max_steps": 10, "shuffle": False, "communication": True}
    )
    obs, state = env_on.reset(key)
    assert jnp.array_equal(obs["agent_0"], obs["agent_1"])  # no message sent yet

    key, subkey = jax.random.split(key)
    obs, state, _, _, _ = env_on.step(
        subkey, state, {"agent_0": jnp.int32(0), "agent_1": jnp.int32(3)}
    )
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
