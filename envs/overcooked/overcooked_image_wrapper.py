"""Image-based observation wrapper for Overcooked.

Replaces the 26-channel symbolic observation with a flat vector containing:
  - RGB image from render_state, normalized to [0,1], flattened
  - 6 scalar values: ego_dir, ego_x, ego_y, partner_dir, partner_x, partner_y

Both agents see the same image but with swapped ego/partner scalars.
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


class OvercookedImageWrapper(BaseEnv):
    """Wrapper that provides image-based observations for Overcooked.

    Observation per agent: concat(image.flatten(), ego_dir, ego_x, ego_y,
                                   partner_dir, partner_x, partner_y)
    where image is (H*tile_size, W*tile_size, 3) float32 in [0,1].
    """

    NUM_SCALARS = 6

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
        self._obs_dim = self._img_flat_dim + self.NUM_SCALARS

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}

        self.agent_view_size = self.env.agent_view_size

    def observation_space(self, agent: str):
        return spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str):
        return self.env.action_space()

    def _make_obs(self, env_state: OvercookedState) -> Dict[str, jnp.ndarray]:
        """Render image and pack per-agent observations."""
        img = render_state(env_state)  # (H*7, W*7, 3) uint8
        img_flat = img.flatten().astype(jnp.float32) / 255.0

        # Per-agent scalars: (dir_idx, pos_x, pos_y)
        # agent_pos is (num_agents, 2) where [i] = (x, y)
        # agent_dir_idx is (num_agents,)
        dir_idx = env_state.agent_dir_idx.astype(jnp.float32)
        pos = env_state.agent_pos.astype(jnp.float32)

        # Agent 0: ego=0, partner=1
        scalars_0 = jnp.array([
            dir_idx[0], pos[0, 0], pos[0, 1],
            dir_idx[1], pos[1, 0], pos[1, 1],
        ])
        # Agent 1: ego=1, partner=0
        scalars_1 = jnp.array([
            dir_idx[1], pos[1, 0], pos[1, 1],
            dir_idx[0], pos[0, 0], pos[0, 1],
        ])

        obs_0 = jnp.concatenate([img_flat, scalars_0])
        obs_1 = jnp.concatenate([img_flat, scalars_1])

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
