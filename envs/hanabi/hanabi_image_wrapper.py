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
from envs.hanabi.hanabi_wrapper import HanabiWrapper, _hanabi_metrics
from envs.hanabi.rendering import (
    IMG_H,
    IMG_W,
    SYMBOLIC_OBS_SIZE,
    SYMBOLIC_ROWS,
    SYMBOLIC_Y0,
    render_hanabi,
)


def _validate_default_two_player_hanabi(env):
    """The image renderer hard-codes the canonical 2P Hanabi geometry."""
    if env.num_agents != 2:
        raise ValueError("HanabiImageWrapper currently supports only num_agents=2.")
    if env.num_colors != 5 or env.num_ranks != 5 or env.hand_size != 5:
        raise ValueError(
            "HanabiImageWrapper currently supports only the default "
            "5-colour, 5-rank, 5-card-hand Hanabi setup."
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
        # The renderer has mixed-height bands, so expose exact pixels via
        # tile_size=1 for agents.initialize_agents._get_image_dims.
        self.grid_height = IMG_H
        self.grid_width = IMG_W
        self.tile_size = 1
        self.num_scalar_obs = 0
        self._img_h = IMG_H
        self._img_w = IMG_W
        self._obs_dim = IMG_H * IMG_W * 3

        super().__init__(*args, **kwargs)
        _validate_default_two_player_hanabi(self.env)

    def observation_space(self, agent: str):
        return jaxmarl_spaces.Box(0.0, 1.0, (self._obs_dim,))

    @staticmethod
    def _render_symbolic_panel(img: jnp.ndarray, symbolic_obs: jnp.ndarray) -> jnp.ndarray:
        """Overlay the exact JaxMARL symbolic obs as a binary pixel panel."""
        padded_len = SYMBOLIC_ROWS * IMG_W
        bits = jnp.pad(
            symbolic_obs.astype(jnp.float32),
            (0, padded_len - SYMBOLIC_OBS_SIZE),
        )
        panel = bits.reshape(SYMBOLIC_ROWS, IMG_W)
        panel_rgb = jnp.broadcast_to(panel[:, :, None] * 255.0, (SYMBOLIC_ROWS, IMG_W, 3))
        panel_rgb = panel_rgb.astype(jnp.uint8)
        return jax.lax.dynamic_update_slice(img, panel_rgb, (SYMBOLIC_Y0, 0, 0))

    def _make_image_obs(
        self,
        env_state,
        symbolic_obs: Optional[Dict[str, jnp.ndarray]] = None,
        old_env_state=None,
        action=None,
        show_last_action=False,
    ) -> Dict[str, jnp.ndarray]:
        """Render per-agent images and flatten to float32 in [0, 1]."""
        obs = {}
        for i, agent in enumerate(self.agents):
            img = render_hanabi(
                env_state,
                jnp.int32(i),
                old_state=old_env_state,
                action=action,
                show_last_action=show_last_action,
            )
            if symbolic_obs is not None:
                img = self._render_symbolic_panel(img, symbolic_obs[agent])
            obs[agent] = img.flatten().astype(jnp.float32) / 255.0
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        symbolic_obs, env_state = self.env.reset(key)
        img_obs = self._make_image_obs(env_state, symbolic_obs=symbolic_obs)
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
        old_env_state = state.env_state
        actor_idx = jnp.nonzero(old_env_state.cur_player_idx, size=1)[0][0]
        action_array = jnp.array([actions[agent] for agent in self.agents])
        acting_action = action_array[actor_idx].astype(jnp.int32)

        symbolic_obs, env_state, rewards, dones, infos = self.env.step(
            key, state.env_state, actions, reset_env_state,
        )

        base_reward = jnp.array([rewards[agent] for agent in self.agents])
        base_return_so_far = base_reward + state.base_return_so_far
        hanabi_metrics = _hanabi_metrics(env_state, actions, self.agents, self.num_agents)
        new_info = {
            **infos,
            "base_return": base_return_so_far,
            "base_reward": base_reward,
            **hanabi_metrics,
        }
        base_return_so_far = jax.lax.select(
            dones["__all__"], jnp.zeros(self.num_agents), base_return_so_far,
        )

        img_obs = self._make_image_obs(
            env_state,
            symbolic_obs=symbolic_obs,
            old_env_state=old_env_state,
            action=acting_action,
            show_last_action=~dones["__all__"],
        )
        avail_actions = self.env.get_legal_moves(env_state)
        new_state = WrappedEnvState(
            env_state=env_state,
            base_return_so_far=base_return_so_far,
            avail_actions=avail_actions,
            step=env_state.turn,
        )
        return img_obs, new_state, rewards, dones, new_info
