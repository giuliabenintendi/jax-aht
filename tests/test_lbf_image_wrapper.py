"""Tests for LBF image observation wrapper and JAX renderer."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from PIL import Image

from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.lbf.rendering.lbf_rendering import (
    render_lbf_state, TILE_PIXELS, _level_color, _FOOD_BASE_COLOR,
    _AGENT_BASE_COLORS, _EMPTY_TILE, _SQUARE, _DIAMOND,
)


GRID_SIZE = 7
NUM_AGENTS = 2
NUM_FOOD = 3


@pytest.fixture
def env():
    return make_env('lbf-image', env_kwargs={
        'grid_size': GRID_SIZE,
        'num_agents': NUM_AGENTS,
        'num_food': NUM_FOOD,
        'max_agent_level': 3,
        'force_coop': True,
    })


@pytest.fixture
def jumanji_env(env):
    """The underlying Jumanji env."""
    return env.env


# --- Renderer tests ---

class TestRenderer:
    def test_empty_tile_shape(self):
        assert _EMPTY_TILE.shape == (TILE_PIXELS, TILE_PIXELS, 3)

    def test_empty_tile_has_grid_lines(self):
        # First row/col pixel should be white grid line
        assert jnp.all(_EMPTY_TILE[0, 0, :] == 255)
        # Interior should be black
        assert jnp.all(_EMPTY_TILE[3, 3, :] == 0)

    def test_shape_masks(self):
        assert _SQUARE.shape == (TILE_PIXELS, TILE_PIXELS)
        assert _DIAMOND.shape == (TILE_PIXELS, TILE_PIXELS)
        # Center pixel should be inside both shapes
        assert _SQUARE[TILE_PIXELS // 2, TILE_PIXELS // 2]
        assert _DIAMOND[TILE_PIXELS // 2, TILE_PIXELS // 2]
        # Corner should be outside both
        assert not _SQUARE[0, 0]
        assert not _DIAMOND[0, 0]
        # Shapes should differ (rect is 5x5, circle has cut corners)
        assert not jnp.array_equal(_SQUARE, _DIAMOND)

    def test_level_color_intensity(self):
        low = _level_color(_FOOD_BASE_COLOR, 1, 5)
        high = _level_color(_FOOD_BASE_COLOR, 5, 5)
        # Higher level = more saturated = further from black = higher mean value
        assert float(jnp.mean(low.astype(jnp.float32))) < float(jnp.mean(high.astype(jnp.float32)))

    def test_level_color_clamp(self):
        # Level 0 should still produce visible color (clamped to t=0.3)
        color = _level_color(_FOOD_BASE_COLOR, 0, 5)
        assert not jnp.all(color == 0)  # not pure black

    def test_render_state_shape(self, jumanji_env):
        state, _ = jumanji_env.reset(jax.random.PRNGKey(0))
        img = render_lbf_state(state, GRID_SIZE, NUM_AGENTS, NUM_FOOD)
        expected_h = GRID_SIZE * TILE_PIXELS
        expected_w = GRID_SIZE * TILE_PIXELS
        assert img.shape == (expected_h, expected_w, 3)
        assert img.dtype == jnp.uint8

    def test_render_state_not_all_black(self, jumanji_env):
        state, _ = jumanji_env.reset(jax.random.PRNGKey(0))
        img = render_lbf_state(state, GRID_SIZE, NUM_AGENTS, NUM_FOOD)
        # Should have non-black pixels (grid lines, agents, food)
        assert not jnp.all(img == 0)

    def test_render_state_jittable(self, jumanji_env):
        state, _ = jumanji_env.reset(jax.random.PRNGKey(0))
        jitted = jax.jit(render_lbf_state, static_argnums=(1, 2, 3, 4, 5))
        img = jitted(state, GRID_SIZE, NUM_AGENTS, NUM_FOOD, 3, 6)
        assert img.shape == (GRID_SIZE * TILE_PIXELS, GRID_SIZE * TILE_PIXELS, 3)

    def test_eaten_food_not_rendered(self, jumanji_env):
        state, _ = jumanji_env.reset(jax.random.PRNGKey(0))
        img_before = render_lbf_state(state, GRID_SIZE, NUM_AGENTS, NUM_FOOD)

        # Mark all food as eaten
        new_eaten = jnp.ones(NUM_FOOD, dtype=jnp.bool_)
        new_food = state.food_items.replace(eaten=new_eaten)
        state_eaten = state.replace(food_items=new_food)
        img_after = render_lbf_state(state_eaten, GRID_SIZE, NUM_AGENTS, NUM_FOOD)

        # Images should differ (food tiles are now empty)
        assert not jnp.array_equal(img_before, img_after)


# --- Image wrapper tests ---

class TestLBFImageWrapper:
    def test_env_creation(self, env):
        assert env.num_agents == NUM_AGENTS
        assert len(env.agents) == NUM_AGENTS
        assert env.grid_height == GRID_SIZE
        assert env.grid_width == GRID_SIZE
        assert env.tile_size == TILE_PIXELS

    def test_observation_space(self, env):
        obs_space = env.observation_space(env.agents[0])
        expected_dim = GRID_SIZE * TILE_PIXELS * GRID_SIZE * TILE_PIXELS * 3
        assert obs_space.shape == (expected_dim,)

    def test_reset_obs_shape(self, env):
        obs, state = env.reset(jax.random.PRNGKey(0))
        expected_dim = GRID_SIZE * TILE_PIXELS * GRID_SIZE * TILE_PIXELS * 3
        for agent in env.agents:
            assert obs[agent].shape == (expected_dim,)
            assert obs[agent].dtype == jnp.float32
            # Values should be in [0, 1]
            assert float(jnp.min(obs[agent])) >= 0.0
            assert float(jnp.max(obs[agent])) <= 1.0

    def test_obs_differ_between_agents(self, env):
        """Each agent should see a different ego border."""
        obs, _ = env.reset(jax.random.PRNGKey(0))
        assert not jnp.array_equal(obs["agent_0"], obs["agent_1"])

    def test_step(self, env):
        obs, state = env.reset(jax.random.PRNGKey(0))
        actions = {agent: 0 for agent in env.agents}  # noop
        obs, state, rewards, dones, info = env.step(
            jax.random.PRNGKey(1), state, actions,
        )
        for agent in env.agents:
            assert obs[agent].shape == env.observation_space(agent).shape
            assert agent in rewards
            assert agent in dones

    def test_avail_actions(self, env):
        _, state = env.reset(jax.random.PRNGKey(0))
        avail = env.get_avail_actions(state)
        for agent in env.agents:
            assert avail[agent].shape == (6,)
            assert jnp.all(avail[agent] == 1.0)

    def test_with_log_wrapper(self, env):
        """Ensure LogWrapper works on top of image wrapper."""
        wrapped = LogWrapper(env)
        obs, state = wrapped.reset(jax.random.PRNGKey(0))
        actions = {agent: 0 for agent in wrapped.agents}
        obs, state, rewards, dones, info = wrapped.step(
            jax.random.PRNGKey(1), state, actions,
        )
        assert "returned_episode_returns" in info
        assert "returned_episode" in info

    def test_multi_step_episode(self, env):
        """Run several steps and verify shapes remain consistent."""
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        for _ in range(10):
            key, step_key, act_key = jax.random.split(key, 3)
            actions = {agent: jax.random.randint(act_key, (), 0, 6)
                       for agent in env.agents}
            obs, state, rewards, dones, info = env.step(step_key, state, actions)
            for agent in env.agents:
                assert obs[agent].shape == env.observation_space(agent).shape


# --- make_env integration test ---

class TestMakeEnv:
    def test_make_env_lbf_image(self):
        env = make_env('lbf-image', env_kwargs={
            'grid_size': 8,
            'num_agents': 2,
            'num_food': 2,
            'force_coop': True,
        })
        assert env.grid_height == 8
        obs, _ = env.reset(jax.random.PRNGKey(0))
        expected_dim = 8 * TILE_PIXELS * 8 * TILE_PIXELS * 3
        assert obs["agent_0"].shape == (expected_dim,)

    def test_make_env_lbf_obs_type_image(self):
        """obs_type='image' in ENV_KWARGS should also produce image wrapper."""
        env = make_env('lbf', env_kwargs={
            'obs_type': 'image',
            'grid_size': 8,
            'num_agents': 2,
            'num_food': 2,
        })
        assert hasattr(env, 'grid_height')
        obs, _ = env.reset(jax.random.PRNGKey(0))
        expected_dim = 8 * TILE_PIXELS * 8 * TILE_PIXELS * 3
        assert obs["agent_0"].shape == (expected_dim,)


class TestSaveImageObs:
    def test_save_image_obs(self):
        """Save sample LBF image observations as PNGs for visual inspection.

        Run with: pytest -s tests/test_lbf_image_wrapper.py::TestSaveImageObs
        """
        env = make_env('lbf-image', env_kwargs={
            'grid_size': GRID_SIZE,
            'num_agents': NUM_AGENTS,
            'num_food': NUM_FOOD,
            'max_agent_level': 3,
            'force_coop': True,
        })
        obs, _ = env.reset(jax.random.PRNGKey(0))

        h = env.grid_height * TILE_PIXELS
        w = env.grid_width * TILE_PIXELS
        scale = 10

        for agent in env.agents:
            arr = np.array(obs[agent])
            img = (arr * 255).astype(np.uint8).reshape(h, w, 3)
            img_large = np.kron(img, np.ones((scale, scale, 1))).astype(np.uint8)
            fname = f"lbf_image_obs_{agent}.png"
            Image.fromarray(img_large).save(fname)
            print(f"\nSaved {fname} ({img_large.shape[0]}x{img_large.shape[1]})")
