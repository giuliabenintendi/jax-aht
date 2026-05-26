"""Image-observation wrapper for Hanabi.

Subclasses HanabiWrapper to keep the inner env, the avail_actions plumbing,
and the base-return tracking, but replaces the symbolic observation dict
with per-agent rendered images from `envs.hanabi.rendering.render_hanabi`.

Exposes `grid_height`, `grid_width`, `tile_size`, `num_scalar_obs` so
`agents.initialize_agents._get_image_dims` can resolve `(img_h, img_w)`
the same way it does for the card-game env.
"""
from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import WrappedEnvState
from envs.hanabi.hanabi_wrapper import HanabiWrapper
from envs.hanabi.rendering import (
    GRID_COLS,
    GRID_ROWS,
    IMG_H,
    IMG_W,
    TILE_PIXELS,
    render_hanabi,
)


class HanabiImageWrapper(HanabiWrapper):
    """Hanabi wrapper emitting flattened per-agent image observations.

    The image is the renderer's egocentric view: partner hand, fireworks,
    info+life token strip, own hand. Returned as a flat `float32` vector
    in `[0, 1]`.
    """

    def __init__(self, *args, **kwargs):
        # Set image-wrapper constants before super().__init__ so the parent's
        # observation_spaces dict-comp (which calls our overridden
        # observation_space) sees the image obs shape, not the symbolic one.
        self.grid_height = GRID_ROWS
        self.grid_width = GRID_COLS
        self.tile_size = TILE_PIXELS
        self.num_scalar_obs = 0
        self._img_h = IMG_H
        self._img_w = IMG_W
        self._obs_dim = IMG_H * IMG_W * 3

        super().__init__(*args, **kwargs)

    def observation_space(self, agent: str):
        return jaxmarl_spaces.Box(0.0, 1.0, (self._obs_dim,))

    def _make_image_obs(self, env_state) -> Dict[str, jnp.ndarray]:
        """Render per-agent images and flatten to float32 in [0, 1]."""
        obs = {}
        for i, agent in enumerate(self.agents):
            img = render_hanabi(env_state, jnp.int32(i))
            obs[agent] = img.flatten().astype(jnp.float32) / 255.0
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        _, env_state = self.env.reset(key)
        img_obs = self._make_image_obs(env_state)
        avail_actions = self.env.get_legal_moves(env_state)
        return img_obs, WrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(self.num_agents),
            avail_actions=avail_actions,
            step=env_state.turn,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        reset_env_state = reset_state.env_state if reset_state is not None else None
        _, env_state, rewards, dones, infos = self.env.step(
            key, state.env_state, actions, reset_env_state,
        )

        base_reward = jnp.array([rewards[agent] for agent in self.agents])
        base_return_so_far = base_reward + state.base_return_so_far
        new_info = {**infos, "base_return": base_return_so_far, "base_reward": base_reward}
        base_return_so_far = jax.lax.select(
            dones["__all__"], jnp.zeros(self.num_agents), base_return_so_far,
        )

        img_obs = self._make_image_obs(env_state)
        avail_actions = self.env.get_legal_moves(env_state)
        new_state = WrappedEnvState(
            env_state=env_state,
            base_return_so_far=base_return_so_far,
            avail_actions=avail_actions,
            step=env_state.turn,
        )
        return img_obs, new_state, rewards, dones, new_info
