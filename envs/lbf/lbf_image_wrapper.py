"""Image-based observation wrapper for Level Based Foraging."""
from typing import Dict

import jax
import jax.numpy as jnp
from jumanji.env import Environment as JumanjiEnv
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.lbf.lbf_wrapper import LBFWrapper
from envs.lbf.rendering.lbf_rendering import render_lbf_state, TILE_PIXELS

_EGO_HIGHLIGHT_COLOR = jnp.array([255, 0, 255], dtype=jnp.uint8)


def _draw_border(img, row, col, tile_size, color):
    """Draw a 1-pixel border around the tile at grid position (row, col)."""
    y = jnp.int32(row) * tile_size
    x = jnp.int32(col) * tile_size

    top_row = jnp.broadcast_to(color, (1, tile_size, 3))
    img = jax.lax.dynamic_update_slice(img, top_row, (y, x, 0))
    img = jax.lax.dynamic_update_slice(img, top_row, (y + tile_size - 1, x, 0))

    left_col = jnp.broadcast_to(color, (tile_size, 1, 3))
    img = jax.lax.dynamic_update_slice(img, left_col, (y, x, 0))
    img = jax.lax.dynamic_update_slice(img, left_col, (y, x + tile_size - 1, 0))

    return img


class LBFImageWrapper(LBFWrapper):
    """LBF wrapper variant that replaces symbolic observations with RGB images.

    Each agent sees the full rendered grid with a magenta border drawn
    around its own tile. Observation is a flat float32 vector.

    Exposes grid_height, grid_width, tile_size for compatibility with
    the JA-IPPO agent initialization pipeline.
    """

    def __init__(self, env: JumanjiEnv, share_rewards: bool = True, **kwargs):
        super().__init__(env, share_rewards=share_rewards)

        # Extract grid params from generator
        self._grid_size = self.env._generator.grid_size
        self._num_food = self.env._generator.num_food
        self._max_agent_level = self.env._generator.max_agent_level
        self._max_food_level = self._max_agent_level * self.num_agents

        # Image dimensions (exposed for initialize_agents._get_image_dims)
        self.grid_height = self._grid_size
        self.grid_width = self._grid_size
        self.tile_size = TILE_PIXELS

        self._img_h = self.grid_height * self.tile_size
        self._img_w = self.grid_width * self.tile_size
        self._img_flat_dim = self._img_h * self._img_w * 3
        self._obs_dim = self._img_flat_dim

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}

    def observation_space(self, agent: str):
        return jaxmarl_spaces.Box(0.0, 1.0, (self._obs_dim,))

    def _render(self, env_state):
        return render_lbf_state(
            env_state, self._grid_size, self.num_agents, self._num_food,
            self._max_agent_level, self._max_food_level,
        )

    def _extract_observations(self, observation, env_state=None) -> Dict[str, jnp.ndarray]:
        """Render image with per-agent ego highlight."""
        del observation
        img = self._render(env_state)

        obs = {}
        for i in range(self.num_agents):
            row = env_state.agents.position[i, 0]
            col = env_state.agents.position[i, 1]
            agent_img = _draw_border(img, row, col, self.tile_size, _EGO_HIGHLIGHT_COLOR)
            obs[self.agents[i]] = agent_img.flatten().astype(jnp.float32) / 255.0
        return obs
