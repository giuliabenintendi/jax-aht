"""FOV image observation wrapper for Overcooked.

Each agent gets an egocentric RGB crop around itself, rotated so its
forward direction faces up. No scalars are appended — position and
direction are implicit in the egocentric view.

Observation per agent: flat float32 vector of length fov_px * fov_px * 3,
where fov_px = fov_size * tile_size (default 7 * 7 = 49).
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


def _crop_fov(
    image: jnp.ndarray,
    agent_pos_xy: jnp.ndarray,
    fov_size: int,
    tile_size: int,
    centered: bool,
) -> jnp.ndarray:
    """Crop a local FOV window around an agent from the full rendered image.

    Args:
        image: (H_px, W_px, 3) full rendered image.
        agent_pos_xy: (2,) agent grid position (x, y).
        fov_size: FOV radius in grid cells (total window = fov_size * tile_size px).
        tile_size: pixels per grid cell.
        centered: if True, agent is at center of crop; if False, agent is at
            bottom-center (sees more ahead than behind).

    Returns:
        (fov_px, fov_px, 3) cropped image, zero-padded where off the map.
    """
    fov_px = fov_size * tile_size
    # Agent pixel center
    ax = agent_pos_xy[0] * tile_size + tile_size // 2
    ay = agent_pos_xy[1] * tile_size + tile_size // 2

    if centered:
        top = ay - fov_px // 2
        left = ax - fov_px // 2
    else:
        # Bottom-aligned: agent near bottom of crop, sees more ahead (up)
        top = ay - fov_px + tile_size // 2 + tile_size
        left = ax - fov_px // 2

    # Use dynamic_slice with zero-padding via pad + slice
    h, w = image.shape[0], image.shape[1]
    pad_top = fov_px
    pad_left = fov_px
    padded = jnp.pad(
        image,
        ((pad_top, pad_top), (pad_left, pad_left), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    # Shift indices to account for padding
    crop = jax.lax.dynamic_slice(
        padded,
        (top + pad_top, left + pad_left, 0),
        (fov_px, fov_px, 3),
    )
    return crop


def _rotate_forward_up(
    crop: jnp.ndarray, dir_idx: jnp.ndarray
) -> jnp.ndarray:
    """Rotate crop so agent's forward direction faces up.

    Direction convention: 0=N(up), 1=S(down), 2=E(right), 3=W(left).
    N already up -> 0 rot. S -> 2 (180°). E -> 1 (CCW 90°). W -> 3 (CCW 270°).
    """
    n_rot = jnp.array([0, 2, 1, 3])[dir_idx]
    # jnp.rot90 with k argument; use lax.switch for JIT compatibility
    def rot0(c):
        return c
    def rot1(c):
        return jnp.rot90(c, k=1)
    def rot2(c):
        return jnp.rot90(c, k=2)
    def rot3(c):
        return jnp.rot90(c, k=3)
    return jax.lax.switch(n_rot, [rot0, rot1, rot2, rot3], crop)


def fov_observation(
    image: jnp.ndarray,
    agent_pos_xy: jnp.ndarray,
    agent_dir_idx: jnp.ndarray,
    fov_size: int,
    tile_size: int,
    centered: bool,
) -> jnp.ndarray:
    """Crop + rotate: egocentric FOV observation for one agent.

    Returns: (fov_px, fov_px, 3) uint8 image, forward=up.
    """
    crop = _crop_fov(image, agent_pos_xy, fov_size, tile_size, centered)
    return _rotate_forward_up(crop, agent_dir_idx)


class OvercookedFOVWrapper(BaseEnv):
    """Wrapper providing egocentric FOV image observations.

    Each agent sees a local (fov_px, fov_px, 3) crop around itself, rotated
    so forward=up. Observations are flattened to 1D float32 in [0,1].
    No additional scalars — position/direction are implicit in the view.
    """

    def __init__(self, *args, fov_size: int = 7, fov_centered: bool = False, **kwargs):
        self.env = OvercookedV1(*args, **kwargs)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)

        self.fov_size = fov_size
        self.fov_centered = fov_centered
        self.tile_size = TILE_PIXELS

        self.fov_px = fov_size * self.tile_size
        self._obs_dim = self.fov_px * self.fov_px * 3

        self.grid_height = self.env.height
        self.grid_width = self.env.width

        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.action_space(a) for a in self.agents}

        self.agent_view_size = self.env.agent_view_size

    def observation_space(self, agent: str):
        return spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str):
        return self.env.action_space()

    def _make_obs(self, env_state: OvercookedState) -> Dict[str, jnp.ndarray]:
        """Render full image, then crop per-agent FOV."""
        img = render_state(env_state)  # (H_px, W_px, 3) uint8

        def agent_fov(agent_idx):
            return fov_observation(
                img,
                env_state.agent_pos[agent_idx],
                env_state.agent_dir_idx[agent_idx],
                self.fov_size,
                self.tile_size,
                self.fov_centered,
            )

        fov_0 = agent_fov(0)
        fov_1 = agent_fov(1)

        obs_0 = fov_0.flatten().astype(jnp.float32) / 255.0
        obs_1 = fov_1.flatten().astype(jnp.float32) / 255.0

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
