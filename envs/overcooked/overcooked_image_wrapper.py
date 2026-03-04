"""Image-based observation wrapper for Overcooked.

Replaces the 26-channel symbolic observation with a flat RGB image.
Each agent sees the full grid with a magenta border drawn around its
own tile, making the two observations distinct without appending scalars.
"""
from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked.overcooked import State as OvercookedState
from jaxmarl.environments import spaces

from envs.overcooked.overcooked_v1 import OvercookedV1
from envs.overcooked.rendering import render_state
from envs.overcooked.rendering.overcooked_rendering import TILE_PIXELS
from envs.base_env import BaseEnv, WrappedEnvState

# Magenta, distinct from all Overcooked tile colors
_EGO_HIGHLIGHT_COLOR = jnp.array([255, 0, 255], dtype=jnp.uint8)


def _draw_border(
    img: jnp.ndarray,
    pos_xy: jnp.ndarray,
    tile_size: int,
    color: jnp.ndarray,
) -> jnp.ndarray:
    """Draw a 1-pixel border around the tile at grid position pos_xy.

    Uses dynamic_update_slice to overwrite border pixels in-place.
    """
    x = jnp.int32(pos_xy[0]) * tile_size
    y = jnp.int32(pos_xy[1]) * tile_size

    # Top edge
    top_row = jnp.broadcast_to(color, (1, tile_size, 3))
    img = jax.lax.dynamic_update_slice(img, top_row, (y, x, 0))
    # Bottom edge
    img = jax.lax.dynamic_update_slice(img, top_row, (y + tile_size - 1, x, 0))
    # Left edge
    left_col = jnp.broadcast_to(color, (tile_size, 1, 3))
    img = jax.lax.dynamic_update_slice(img, left_col, (y, x, 0))
    # Right edge
    img = jax.lax.dynamic_update_slice(img, left_col, (y, x + tile_size - 1, 0))

    return img


class OvercookedImageWrapper(BaseEnv):
    """Wrapper that provides image-based observations for Overcooked.

    Each agent sees the full rendered grid with a magenta border drawn
    around its own tile. Observation is a flat float32 vector (image only,
    no appended scalars).
    """

    def __init__(self, *args, **kwargs):
        self.env = OvercookedV1(*args, **kwargs)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)

        self.grid_height = self.env.height
        self.grid_width = self.env.width
        self.tile_size = TILE_PIXELS

        self._img_h = self.grid_height * self.tile_size
        self._img_w = self.grid_width * self.tile_size
        self._img_flat_dim = self._img_h * self._img_w * 3
        self._obs_dim = self._img_flat_dim

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}

        self.agent_view_size = self.env.agent_view_size

    def observation_space(self, agent: str):
        return spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str):
        return self.env.action_space()

    def _make_obs(self, env_state: OvercookedState) -> Dict[str, jnp.ndarray]:
        """Render image with per-agent ego highlight."""
        img = render_state(env_state)  # (H*7, W*7, 3) uint8

        img_0 = _draw_border(img, env_state.agent_pos[0], self.tile_size, _EGO_HIGHLIGHT_COLOR)
        img_1 = _draw_border(img, env_state.agent_pos[1], self.tile_size, _EGO_HIGHLIGHT_COLOR)

        obs_0 = img_0.flatten().astype(jnp.float32) / 255.0
        obs_1 = img_1.flatten().astype(jnp.float32) / 255.0

        return {"agent_0": obs_0, "agent_1": obs_1}

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        _, env_state = self.env.reset(key)
        obs = self._make_obs(env_state)
        return obs, WrappedEnvState(
            env_state, jnp.zeros(self.num_agents),
            jnp.zeros(self.num_agents), jnp.empty((), dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        num_actions = len(self.env.action_set)
        return {agent: jnp.ones(num_actions) for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        return state.env_state.time

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        obs, env_state, rewards, dones, infos = self.env.step(key, state.env_state, actions, reset_state)
        obs = self._make_obs(env_state)

        base_reward = infos['base_reward']
        base_return_so_far = base_reward + state.base_return_so_far
        new_info = {**infos, 'base_return': base_return_so_far}

        base_return_so_far = jax.lax.select(
            dones['__all__'], jnp.zeros(self.num_agents), base_return_so_far,
        )
        new_state = WrappedEnvState(
            env_state=env_state, base_return_so_far=base_return_so_far,
            avail_actions=jnp.zeros(self.num_agents),
            step=jnp.empty((), dtype=jnp.int32),
        )
        return obs, new_state, rewards, dones, new_info
