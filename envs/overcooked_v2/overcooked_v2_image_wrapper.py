"""Image-based observation wrapper for Overcooked V2.

Overcooked V2 is partially observable: each agent sees a square view radius
around itself. This wrapper renders the full god's-eye grid and then, per
agent, zeroes out every cell outside the agent's view box (allocentric
view-masking) and draws a magenta border around the agent's own tile. Keeping
a shared allocentric frame (rather than an egocentric crop) means both agents'
observations live in the same spatial coordinates, which the joint-attention
machinery relies on.

The underlying env returns a sparse delivery reward plus a separate per-agent
shaped reward in `info["shaped_reward"]`. When `do_reward_shaping` is set the
shaped term is folded into the agent reward used for learning, while the
sparse delivery return is tracked separately in `base_return_so_far`.

Other-Play: when the base env is built with `op_ingredient_permutations`, each
agent's view is rendered with its own permuted ingredient palette (see
`_agent_palettes`), reproducing the paper's per-agent observation relabeling.
Evaluation with non-permuted observations = build the env without the kwarg.
"""
from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments import spaces

from envs.base_env import BaseEnv, WrappedEnvState
from envs.overcooked_v2.overcooked import OvercookedV2
from envs.overcooked_v2.rendering import INGREDIENT_COLORS, TILE_PIXELS, render_state
from envs.overcooked_v2.utils import compute_view_box

# Magenta, distinct from all Overcooked tile colors
_EGO_HIGHLIGHT_COLOR = jnp.array([255, 0, 255], dtype=jnp.uint8)


def _draw_border(
    img: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    tile_size: int,
    color: jnp.ndarray,
) -> jnp.ndarray:
    """Draw a 1-pixel border around the tile at grid cell (x, y)."""
    px = jnp.int32(x) * tile_size
    py = jnp.int32(y) * tile_size

    top_row = jnp.broadcast_to(color, (1, tile_size, 3))
    img = jax.lax.dynamic_update_slice(img, top_row, (py, px, 0))
    img = jax.lax.dynamic_update_slice(img, top_row, (py + tile_size - 1, px, 0))

    left_col = jnp.broadcast_to(color, (tile_size, 1, 3))
    img = jax.lax.dynamic_update_slice(img, left_col, (py, px, 0))
    img = jax.lax.dynamic_update_slice(img, left_col, (py, px + tile_size - 1, 0))

    return img


class OvercookedV2ImageWrapper(BaseEnv):
    """Allocentric view-masked RGB observations for Overcooked V2."""

    def __init__(
        self,
        tile_size: int = TILE_PIXELS,
        do_reward_shaping: bool = True,
        **kwargs,
    ):
        self.env = OvercookedV2(**kwargs)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)

        self.grid_height = self.env.height
        self.grid_width = self.env.width
        self.tile_size = tile_size
        # View radius; None means full observability.
        self.agent_view_size = self.env.agent_view_size
        self.do_reward_shaping = do_reward_shaping

        self._img_h = self.grid_height * self.tile_size
        self._img_w = self.grid_width * self.tile_size
        self._obs_dim = self._img_h * self._img_w * 3

        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.action_space(a) for a in self.agents}

    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str = "") -> spaces.Discrete:
        return self.env.action_space(agent)

    def _visibility_mask(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Boolean (grid_height, grid_width) mask of cells in the agent's view."""
        if self.agent_view_size is None:
            return jnp.ones((self.grid_height, self.grid_width), dtype=bool)
        x_low, x_high, y_low, y_high = compute_view_box(
            x, y, self.agent_view_size, self.grid_height, self.grid_width
        )
        rows = jnp.arange(self.grid_height)
        cols = jnp.arange(self.grid_width)
        row_vis = (rows >= y_low) & (rows < y_high)
        col_vis = (cols >= x_low) & (cols < x_high)
        return row_vis[:, None] & col_vis[None, :]

    def _agent_palettes(self, env_state) -> jnp.ndarray:
        """Per-agent ingredient palettes implementing Other-Play in image space.

        The base env samples a per-agent permutation into
        `state.ingredient_permutations`; true ingredient i must be shown with the
        colour of slot perm^{-1}[i] (matching the symbolic `get_obs` relabeling).
        Permuting the palette at render time relabels ALL ingredient pixels —
        piles, pots, dishes, recipe indicator — before anti-aliasing, which a
        post-hoc pixel recolour cannot do.
        """
        perms = env_state.ingredient_permutations  # (num_agents, n_ing)
        n_ing = perms.shape[-1]
        inv = jax.vmap(
            lambda p: jnp.zeros_like(p).at[p].set(jnp.arange(n_ing, dtype=p.dtype))
        )(perms)
        return jax.vmap(
            lambda iv: INGREDIENT_COLORS.at[:n_ing].set(INGREDIENT_COLORS[iv])
        )(inv)

    def _make_obs(self, env_state) -> Dict[str, jnp.ndarray]:
        if self.env.op_ingredient_permutations:
            palettes = self._agent_palettes(env_state)
            imgs = [
                render_state(env_state, self.tile_size, ingredient_colors=palettes[i])
                for i in range(self.num_agents)
            ]
        else:
            img = render_state(env_state, self.tile_size)  # (H_px, W_px, 3) uint8
            imgs = [img] * self.num_agents

        positions = env_state.agents.pos
        obs = {}
        for i in range(self.num_agents):
            x = positions.x[i]
            y = positions.y[i]
            cell_mask = self._visibility_mask(x, y)
            pixel_mask = jnp.repeat(
                jnp.repeat(cell_mask, self.tile_size, axis=0), self.tile_size, axis=1
            )
            masked = jnp.where(pixel_mask[:, :, None], imgs[i], 0)
            masked = _draw_border(masked, x, y, self.tile_size, _EGO_HIGHLIGHT_COLOR)
            obs[self.agents[i]] = masked.flatten().astype(jnp.float32) / 255.0
        return obs

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        _, env_state = self.env.reset(key)
        obs = self._make_obs(env_state)
        return obs, WrappedEnvState(
            env_state,
            jnp.zeros(self.num_agents),
            jnp.zeros(self.num_agents),
            jnp.empty((), dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        num_actions = self.env.action_space().n
        return {agent: jnp.ones(num_actions) for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.ndarray:
        return state.env_state.time

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,  # noqa: ARG002 — auto-reset
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        obs_sym, env_state, rewards, dones, infos = self.env.step(
            key, state.env_state, actions
        )
        del obs_sym
        obs = self._make_obs(env_state)

        shaped = infos["shaped_reward"]
        if self.do_reward_shaping:
            agent_rewards = {a: rewards[a] + shaped[a] for a in self.agents}
        else:
            agent_rewards = rewards

        base_reward = jnp.stack([rewards[a] for a in self.agents])
        base_return_so_far = base_reward + state.base_return_so_far
        # Info leaves must be per-agent arrays (num_agents,) so the trainer can
        # reshape them to (num_actors,); the env's shaped_reward is a per-agent
        # dict of scalars, so stack it rather than passing the dict through.
        new_info = {
            "shaped_reward": jnp.stack([shaped[a] for a in self.agents]),
            "base_return": base_return_so_far,
        }
        base_return_so_far = jax.lax.select(
            dones["__all__"], jnp.zeros(self.num_agents), base_return_so_far
        )

        new_state = WrappedEnvState(
            env_state=env_state,
            base_return_so_far=base_return_so_far,
            avail_actions=jnp.zeros(self.num_agents),
            step=jnp.empty((), dtype=jnp.int32),
        )
        return obs, new_state, agent_rewards, dones, new_info
