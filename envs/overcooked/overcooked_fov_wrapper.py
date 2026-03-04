"""FOV image observation wrapper for Overcooked.

Ported from OGC (ogc/ogc.py). Each agent gets an egocentric RGB crop around
itself, rotated so its forward direction faces up. No scalars are appended —
position and direction are implicit in the egocentric view.

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

# Rotation lookup: direction index -> number of CCW 90° rotations.
# Matches OGC's align_forward_up: k = [2, 1, 0, 3][dir_idx].
_DIR_TO_ROT = jnp.array([2, 1, 0, 3], dtype=jnp.int32)


def _crop_center(
    padded: jnp.ndarray,
    ax: jnp.ndarray,
    ay: jnp.ndarray,
    fov_h_px: int,
    fov_w_px: int,
) -> jnp.ndarray:
    """Center-aligned crop (agent at center of window)."""
    half_h = fov_h_px // 2
    half_w = fov_w_px // 2
    row = jnp.int32(ay - half_h)
    col = jnp.int32(ax - half_w)
    return jax.lax.dynamic_slice(padded, (row, col, jnp.int32(0)), (fov_h_px, fov_w_px, 3))


def _crop_bottom(
    padded: jnp.ndarray,
    ax: jnp.ndarray,
    ay: jnp.ndarray,
    dir_idx: jnp.ndarray,
    fov_h_px: int,
    fov_w_px: int,
    tile_size: int,
) -> jnp.ndarray:
    """Bottom-aligned crop (agent sees more ahead). Direction-dependent offset.

    Matches OGC's crop_field_of_view_3d_bottom: the crop window shifts so
    the agent is near the trailing edge relative to its facing direction.
    """
    half_h = fov_h_px // 2
    half_w = fov_w_px // 2
    half_tile = tile_size // 2

    # Each direction places the agent near the trailing edge of the crop.
    # Coordinates are in the padded image (ax, ay already include padding offset).
    def L_up():
        return (ay - fov_h_px + 1 + half_tile, ax - half_w, jnp.int32(0))
    def L_right():
        return (ay - half_h, ax - half_tile, jnp.int32(0))
    def L_down():
        return (ay - half_tile, ax - half_w, jnp.int32(0))
    def L_left():
        return (ay - half_h, ax - fov_w_px + 1 + half_tile, jnp.int32(0))

    # OGC direction order: N=0->L_up(idx=2), S=1->L_down(idx=1), E=2->L_right(idx=0), W=3->L_left(idx=3)
    # But lax.switch indexes 0,1,2,3 into the branch list, so we reorder:
    # idx = [2,1,0,3][dir_idx] means dir 0(N)->idx 2, dir 1(S)->idx 1, dir 2(E)->idx 0, dir 3(W)->idx 3
    # branch list at [0]=L_right, [1]=L_down, [2]=L_up, [3]=L_left
    idx = _DIR_TO_ROT[dir_idx]
    start = jax.lax.switch(idx, (L_right, L_down, L_up, L_left))
    return jax.lax.dynamic_slice(padded, start, (fov_h_px, fov_w_px, 3))


def _align_forward_up(
    crop: jnp.ndarray, dir_idx: jnp.ndarray
) -> jnp.ndarray:
    """Rotate crop so agent's forward direction faces up (negative y).

    Matches OGC's align_forward_up exactly: k = [2, 1, 0, 3][dir_idx].
    """
    k = _DIR_TO_ROT[dir_idx]

    def rot0(c): return c
    def rot1(c): return jnp.rot90(c, k=1, axes=(0, 1))
    def rot2(c): return jnp.rot90(c, k=2, axes=(0, 1))
    def rot3(c): return jnp.rot90(c, k=3, axes=(0, 1))

    return jax.lax.switch(k, (rot0, rot1, rot2, rot3), crop)


def fov_observation(
    image: jnp.ndarray,
    agent_pos_xy: jnp.ndarray,
    agent_dir_idx: jnp.ndarray,
    fov_size: int,
    tile_size: int,
    centered: bool,
) -> jnp.ndarray:
    """Crop + rotate: egocentric FOV observation for one agent.

    Returns: (fov_px, fov_px, 3) image, forward=up.
    """
    fov_px = fov_size * tile_size

    # Agent pixel position (tile center), cast to int32
    ax = jnp.int32(agent_pos_xy[0]) * tile_size + tile_size // 2
    ay = jnp.int32(agent_pos_xy[1]) * tile_size + tile_size // 2

    # Pad image so crops near edges are zero-filled
    padded = jnp.pad(
        image,
        ((fov_px, fov_px), (fov_px, fov_px), (0, 0)),
        constant_values=0,
    )
    # Shift to padded coordinates
    ax = ax + fov_px
    ay = ay + fov_px

    if centered:
        crop = _crop_center(padded, ax, ay, fov_px, fov_px)
    else:
        crop = _crop_bottom(padded, ax, ay, agent_dir_idx, fov_px, fov_px, tile_size)

    return _align_forward_up(crop, agent_dir_idx)


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
