"""Tests for the OP-test diagnostic env and its position-shuffle wrapper."""
import jax
import jax.numpy as jnp

from envs import make_env
from envs.card_game.action_utils import NO_COMM_NOOP_ACTION
from envs.card_game.card_game_op_test import (
    CardGameOPTestEnv,
    OPTestPositionShuffleWrapper,
    RED_COLOR,
    WHITE_COLOR,
)
from envs.card_game.rendering import NUM_CARDS, TILE_PIXELS


def _get_card_color_at(flat_obs, pos, img_h, img_w):
    """Read the center pixel of the card at view position pos."""
    img = (flat_obs * 255).astype(jnp.uint8).reshape(img_h, img_w, 3)
    TP = TILE_PIXELS
    return img[TP + 3, pos * TP + 3, :]


# 1. Env basics --------------------------------------------------------------

def test_env_action_space_is_six():
    env = make_env("card-game-op-test", {})
    assert env.action_space("agent_0").n == NUM_CARDS + 1


def test_env_obs_has_one_red_four_white():
    """For every reset, exactly one position is RED and the rest are WHITE."""
    env = CardGameOPTestEnv(max_steps=2)
    for seed in range(10):
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset(key)
        red_pos = int(state.env_state.red_position)
        for agent in env.agents:
            for pos in range(NUM_CARDS):
                actual = _get_card_color_at(
                    obs[agent], pos, env._img_h, env._img_w
                )
                expected = RED_COLOR if pos == red_pos else WHITE_COLOR
                assert jnp.all(actual == expected), (
                    f"seed={seed} {agent} pos {pos}: "
                    f"expected {expected}, got {actual}"
                )


def test_red_position_resampled_per_episode():
    """red_position should not be constant across resets."""
    env = CardGameOPTestEnv(max_steps=2)
    seen = set()
    for seed in range(30):
        _, state = env.reset(jax.random.PRNGKey(seed))
        seen.add(int(state.env_state.red_position))
    assert len(seen) > 1, "red_position is constant across resets"


# 2. Reward semantics --------------------------------------------------------

def _run_to_decision(env, key, actions_decision, max_steps=2):
    """Drive an episode through deliberation noops to a decision step."""
    obs, state = env.reset(key)
    noop = NO_COMM_NOOP_ACTION
    for _ in range(max_steps - 1):
        key, subkey = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            subkey, state,
            {"agent_0": jnp.int32(noop), "agent_1": jnp.int32(noop)},
        )
    key, subkey = jax.random.split(key)
    return env.step(subkey, state, actions_decision)


def test_reward_red_match_pays_0_9():
    env = make_env("card-game-op-test", {"max_steps": 2})
    for seed in range(10):
        _, state = env.reset(jax.random.PRNGKey(seed))
        red_pos = int(state.env_state.red_position)
        actions = {
            "agent_0": jnp.int32(red_pos),
            "agent_1": jnp.int32(red_pos),
        }
        _, _, reward, dones, _ = _run_to_decision(
            env, jax.random.PRNGKey(seed), actions, max_steps=2
        )
        assert dones["__all__"]
        assert float(reward["agent_0"]) == 0.9, (
            f"seed {seed}: red_pos={red_pos} got {float(reward['agent_0'])}"
        )


def test_reward_white_match_pays_1_0():
    env = make_env("card-game-op-test", {"max_steps": 2})
    for seed in range(10):
        _, state = env.reset(jax.random.PRNGKey(seed))
        red_pos = int(state.env_state.red_position)
        white_pos = (red_pos + 1) % NUM_CARDS
        actions = {
            "agent_0": jnp.int32(white_pos),
            "agent_1": jnp.int32(white_pos),
        }
        _, _, reward, dones, _ = _run_to_decision(
            env, jax.random.PRNGKey(seed), actions, max_steps=2
        )
        assert dones["__all__"]
        assert float(reward["agent_0"]) == 1.0, (
            f"seed {seed}: white_pos={white_pos} got {float(reward['agent_0'])}"
        )


def test_reward_mismatch_pays_zero():
    env = make_env("card-game-op-test", {"max_steps": 2})
    actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(2)}
    _, _, reward, _, _ = _run_to_decision(
        env, jax.random.PRNGKey(42), actions, max_steps=2
    )
    assert float(reward["agent_0"]) == 0.0


def test_reward_only_on_decision_step():
    """Picks during deliberation must not score (mask should prevent this,
    but the reward gating must hold even if the mask is bypassed)."""
    env = make_env("card-game-op-test", {"max_steps": 4})
    _, state = env.reset(jax.random.PRNGKey(7))
    red_pos = int(state.env_state.red_position)
    # Deliberation step: pick red on both sides — should not score
    _, state, reward, dones, _ = env.step(
        jax.random.PRNGKey(8),
        state,
        {"agent_0": jnp.int32(red_pos), "agent_1": jnp.int32(red_pos)},
    )
    assert not dones["__all__"]
    assert float(reward["agent_0"]) == 0.0


# 3. Position-shuffle wrapper -----------------------------------------------

def test_position_shuffle_obs_swaps_tiles():
    """View position j must show the GT tile at perm[j]."""
    env = OPTestPositionShuffleWrapper(CardGameOPTestEnv(max_steps=8))
    for seed in range(10):
        obs, state = env.reset(jax.random.PRNGKey(seed))
        red_pos_gt = int(state.env_state.env_state.red_position)
        for agent in env.agents:
            perm = state.per_agent_perm[agent]
            for view_pos in range(NUM_CARDS):
                gt_pos = int(perm[view_pos])
                actual = _get_card_color_at(
                    obs[agent], view_pos, env._env._img_h, env._env._img_w
                )
                expected = RED_COLOR if gt_pos == red_pos_gt else WHITE_COLOR
                assert jnp.all(actual == expected), (
                    f"seed={seed} {agent} view {view_pos} (gt {gt_pos}): "
                    f"expected {expected}, got {actual}"
                )


def test_position_shuffle_pick_inversion_via_red_view_match():
    """Both agents picking red 'in their own view' must coordinate on GT red."""
    env = OPTestPositionShuffleWrapper(CardGameOPTestEnv(max_steps=2))
    for seed in range(20):
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset(key)
        red_pos_gt = int(state.env_state.env_state.red_position)

        # In each agent's view, red appears at perm^-1[red_pos_gt]
        view_pick_0 = int(jnp.argmax(
            jnp.equal(state.per_agent_perm["agent_0"], red_pos_gt)
        ))
        view_pick_1 = int(jnp.argmax(
            jnp.equal(state.per_agent_perm["agent_1"], red_pos_gt)
        ))

        key, subkey = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            subkey, state,
            {"agent_0": jnp.int32(NO_COMM_NOOP_ACTION),
             "agent_1": jnp.int32(NO_COMM_NOOP_ACTION)},
        )
        key, subkey = jax.random.split(key)
        _, _, reward, dones, _ = env.step(
            subkey, state,
            {"agent_0": jnp.int32(view_pick_0),
             "agent_1": jnp.int32(view_pick_1)},
        )
        assert dones["__all__"]
        assert float(reward["agent_0"]) == 0.9, (
            f"seed {seed}: red GT={red_pos_gt}, "
            f"view picks ({view_pick_0}, {view_pick_1}), "
            f"reward {float(reward['agent_0'])}"
        )


def test_position_shuffle_same_view_pick_can_mismatch_gt():
    """Without OP-aware coordination, same view position → different GT."""
    env = OPTestPositionShuffleWrapper(CardGameOPTestEnv(max_steps=2))
    found = False
    for seed in range(50):
        key = jax.random.PRNGKey(1000 + seed)
        obs, state = env.reset(key)
        perm_0 = state.per_agent_perm["agent_0"]
        perm_1 = state.per_agent_perm["agent_1"]
        if jnp.all(perm_0 == perm_1):
            continue  # need differing perms
        for view_pos in range(NUM_CARDS):
            if int(perm_0[view_pos]) != int(perm_1[view_pos]):
                key, subkey = jax.random.split(key)
                obs, state, _, _, _ = env.step(
                    subkey, state,
                    {"agent_0": jnp.int32(NO_COMM_NOOP_ACTION),
                     "agent_1": jnp.int32(NO_COMM_NOOP_ACTION)},
                )
                key, subkey = jax.random.split(key)
                _, _, reward, _, _ = env.step(
                    subkey, state,
                    {"agent_0": jnp.int32(view_pos),
                     "agent_1": jnp.int32(view_pos)},
                )
                assert float(reward["agent_0"]) == 0.0, (
                    f"seed {seed}: same view pos {view_pos} mapped to "
                    f"differing GT positions but reward "
                    f"{float(reward['agent_0'])} != 0"
                )
                found = True
                return
    if not found:
        raise AssertionError(
            "Could not find a seed with differing per-agent perms in 50 tries"
        )


def test_position_shuffle_perm_resampled_on_auto_reset():
    """After auto-reset, per-agent perms are freshly drawn."""
    env = OPTestPositionShuffleWrapper(CardGameOPTestEnv(max_steps=2))
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    perm_before = state.per_agent_perm["agent_0"]

    # Drive to end of episode (auto-reset triggers)
    key, subkey = jax.random.split(key)
    _, state, _, _, _ = env.step(
        subkey, state,
        {"agent_0": jnp.int32(NO_COMM_NOOP_ACTION),
         "agent_1": jnp.int32(NO_COMM_NOOP_ACTION)},
    )
    key, subkey = jax.random.split(key)
    _, state, _, dones, _ = env.step(
        subkey, state,
        {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)},
    )
    assert dones["__all__"]
    # State after auto-reset
    perm_after = state.per_agent_perm["agent_0"]
    # With overwhelming probability the perms differ across two random draws
    # (1/120 chance of accidental equality).
    assert not jnp.all(perm_before == perm_after), (
        "Perm should be resampled on auto-reset"
    )


def test_make_env_dispatch_with_op():
    """Factory wires the wrapper when op_position_shuffle=True."""
    env = make_env(
        "card-game-op-test", {"max_steps": 4, "op_position_shuffle": True}
    )
    assert isinstance(env, OPTestPositionShuffleWrapper)


def test_make_env_dispatch_without_op():
    """Factory returns bare env when op_position_shuffle=False."""
    env = make_env(
        "card-game-op-test", {"max_steps": 4, "op_position_shuffle": False}
    )
    assert isinstance(env, CardGameOPTestEnv)
