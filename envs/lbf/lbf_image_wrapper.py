"""Image-based observation wrapper for Level Based Foraging.

Replaces the flat symbolic observation with a flat RGB image rendered from
the full grid state. Each agent sees the full grid with a colored border
drawn around its own tile (magenta), matching the Overcooked image wrapper
interface so the same JA-IPPO pipeline can be reused.
"""
from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from jumanji.env import Environment as JumanjiEnv
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv, WrappedEnvState
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


class LBFImageWrapper(BaseEnv):
    """Wrapper that provides image-based observations for LBF.

    Each agent sees the full rendered grid with a magenta border drawn
    around its own tile. Observation is a flat float32 vector.

    Exposes grid_height, grid_width, tile_size for compatibility with
    the JA-IPPO agent initialization pipeline.
    """

    def __init__(self, env: JumanjiEnv, share_rewards: bool = True, **kwargs):
        self.env = env
        self.share_rewards = share_rewards

        self.num_agents = self.env.num_agents
        self.name = self.env.__class__.__name__
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

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
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}

        # No interior walls in LBF
        self.interior_wall_mask = jnp.zeros((self._grid_size, self._grid_size), dtype=jnp.bool_)

    def observation_space(self, agent: str):
        return jaxmarl_spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str):
        from jaxmarl.environments import spaces as jaxmarl_spaces
        spec = self.env.action_spec
        # LBF has 6 actions: noop, up, right, down, left, load
        num_actions = spec.num_values[0] if hasattr(spec, 'num_values') else 6
        return jaxmarl_spaces.Discrete(num_categories=int(num_actions))

    def _render(self, env_state):
        return render_lbf_state(
            env_state, self._grid_size, self.num_agents, self._num_food,
            self._max_agent_level, self._max_food_level,
        )

    def _make_obs(self, env_state) -> Dict[str, jnp.ndarray]:
        """Render image with per-agent ego highlight."""
        img = self._render(env_state)

        obs = {}
        for i in range(self.num_agents):
            row = env_state.agents.position[i, 0]
            col = env_state.agents.position[i, 1]
            agent_img = _draw_border(img, row, col, self.tile_size, _EGO_HIGHLIGHT_COLOR)
            obs[self.agents[i]] = agent_img.flatten().astype(jnp.float32) / 255.0
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        env_state, timestep = self.env.reset(key)
        obs = self._make_obs(env_state)
        return obs, WrappedEnvState(
            env_state, jnp.zeros(self.num_agents),
            jnp.zeros(self.num_agents), timestep.observation.step_count,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_reset = jax.random.split(key)

        actions_array = jnp.array([actions[agent] for agent in self.agents], dtype=jnp.int32)
        env_state, timestep = self.env.step(state.env_state, actions_array)

        obs_st = self._make_obs(env_state)
        reward = self._extract_rewards(timestep.reward)
        done = timestep.last()
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        base_reward_arr = jnp.array([timestep.reward[i] for i in range(self.num_agents)])
        base_return_so_far = base_reward_arr + state.base_return_so_far

        info = {k: jnp.array([v for _ in range(self.num_agents)])
                for k, v in timestep.extras.items()}
        info['base_reward'] = base_reward_arr
        info['base_return'] = base_return_so_far

        avail_actions = {agent: timestep.observation.action_mask[i]
                         for i, agent in enumerate(self.agents)}

        state_st = WrappedEnvState(
            env_state=env_state,
            base_return_so_far=base_return_so_far,
            avail_actions=jnp.zeros(self.num_agents),
            step=timestep.observation.step_count,
        )

        # Auto-reset on episode end
        obs_reset, state_reset = self.reset(key_reset)
        obs, new_state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y),
            (obs_reset, state_reset),
            (obs_st, state_st),
        )

        # Reset base_return on done
        new_base_return = jax.lax.select(
            done, jnp.zeros(self.num_agents), new_state.base_return_so_far,
        )
        new_state = new_state.replace(base_return_so_far=new_base_return)

        return obs, new_state, reward, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        # Recompute from env state since we don't cache action masks in WrappedEnvState
        # for the image wrapper (avail_actions field stores zeros as placeholder)
        timestep_obs = self.env.observation_spec
        # For LBF, all 6 actions are always structurally available;
        # the env handles invalid moves as no-ops
        num_actions = 6
        return {agent: jnp.ones(num_actions) for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        return state.step

    def _extract_rewards(self, reward):
        if self.share_rewards:
            tot_reward = jnp.mean(reward)
            return {agent: tot_reward for agent in self.agents}
        return {agent: reward[i] for i, agent in enumerate(self.agents)}
