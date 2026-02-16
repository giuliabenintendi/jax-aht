from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked.overcooked import State as OvercookedState
from jaxmarl.environments import spaces

from envs.overcooked.overcooked_v1 import OvercookedV1

from ..base_env import BaseEnv
from ..base_env import WrappedEnvState

class OvercookedWrapper(BaseEnv):
    '''Wrapper for the Overcooked-v1 environment to ensure that it follows a common interface 
    with other environments provided in this library.
    
    Main features:
    - Randomized agent order
    - Flattened observations
    - Base return tracking
    - Optional partial observability with forward cones and occlusion
    '''
    def __init__(
        self,
        *args,
        po_mode: str = "none",          # "none" | "cone"
        fov_range: int = 5,             # max forward distance
        fov_slope: float = 0.7,         # cone width: abs(lateral) <= slope * forward
        use_occlusion: bool = False,    # if True, block visibility behind walls/counters
        soft_view: bool = True,         # if True, apply soft weights inside FOV
        dist_sigma: float = 3.0,        # softness vs distance (bigger = less decay)
        ang_sigma: float = 1.5,         # softness vs off-axis (bigger = less decay)
        **kwargs,
    ):
        if po_mode not in {"none", "cone"}:
            raise ValueError(f"Unsupported po_mode '{po_mode}'. Expected one of: 'none', 'cone'.")

        self.env = OvercookedV1(*args, **kwargs)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)

        self.po_mode = po_mode
        self.fov_range = fov_range
        self.fov_slope = fov_slope
        self.use_occlusion = use_occlusion
        self.soft_view = soft_view
        self.dist_sigma = dist_sigma
        self.ang_sigma = ang_sigma

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}
        
        # exposing some variables from underlying environment
        self.agent_view_size = self.env.agent_view_size

    def observation_space(self, agent: str):
        """Returns the flattened observation space."""
        # Calculate flattened observation shape
        flat_obs_shape = (self.env.obs_shape[0] * self.env.obs_shape[1] * self.env.obs_shape[2],)
        return spaces.Box(0, 255, flat_obs_shape)

    def action_space(self, agent: str):
        return self.env.action_space()

    def _cone_forward_lateral(self, h: int, w: int, pos_xy: jnp.ndarray, dir_idx: jnp.ndarray):
        """Return (forward, lateral) arrays of shape (H, W)."""
        x0, y0 = pos_xy[0], pos_xy[1]
        xs = jnp.arange(w)[None, :]
        ys = jnp.arange(h)[:, None]
        dx = xs - x0
        dy = ys - y0

        # DIR_TO_VEC mapping:
        # 0=NORTH: forward=-dy, lateral= dx
        # 1=SOUTH: forward= dy, lateral= dx
        # 2=EAST:  forward= dx, lateral= dy
        # 3=WEST:  forward=-dx, lateral= dy
        f_n, l_n = -dy, dx
        f_s, l_s = dy, dx
        f_e, l_e = dx, dy
        f_w, l_w = -dx, dy

        forward = jnp.select(
            [dir_idx == 0, dir_idx == 1, dir_idx == 2, dir_idx == 3],
            [f_n, f_s, f_e, f_w],
            default=f_e,
        )
        lateral = jnp.select(
            [dir_idx == 0, dir_idx == 1, dir_idx == 2, dir_idx == 3],
            [l_n, l_s, l_e, l_w],
            default=l_e,
        )
        return forward, lateral

    def _occlusion_mask(self, env_state: OvercookedState, agent_index: int, h: int, w: int) -> jnp.ndarray:
        """Return visibility mask with line-of-sight blocking by wall/counter tiles."""
        pos_xy = env_state.agent_pos[agent_index].astype(jnp.int32)
        wall_map = env_state.wall_map.astype(jnp.bool_)
        x0, y0 = pos_xy[0], pos_xy[1]

        xs = jnp.arange(w, dtype=jnp.int32)[None, :]
        ys = jnp.arange(h, dtype=jnp.int32)[:, None]
        dx = xs - x0
        dy = ys - y0
        steps = jnp.maximum(jnp.abs(dx), jnp.abs(dy)).astype(jnp.int32)
        steps_safe = jnp.maximum(steps, 1)

        max_steps = max(h, w)
        t = jnp.arange(max_steps, dtype=jnp.float32)[:, None, None]
        steps_f = steps_safe.astype(jnp.float32)[None, :, :]

        x_samples = jnp.rint(x0.astype(jnp.float32) + (dx.astype(jnp.float32)[None, :, :] * t / steps_f)).astype(jnp.int32)
        y_samples = jnp.rint(y0.astype(jnp.float32) + (dy.astype(jnp.float32)[None, :, :] * t / steps_f)).astype(jnp.int32)
        x_samples = jnp.clip(x_samples, 0, w - 1)
        y_samples = jnp.clip(y_samples, 0, h - 1)

        valid_steps = (t > 0.0) & (t < steps.astype(jnp.float32)[None, :, :])
        blocked = valid_steps & wall_map[y_samples, x_samples]
        return ~jnp.any(blocked, axis=0)

    def _fov_mask(self, env_state: OvercookedState, agent_index: int, h: int, w: int) -> jnp.ndarray:
        """Boolean visibility mask (H, W)."""
        if self.po_mode == "none":
            return jnp.ones((h, w), dtype=jnp.bool_)

        pos_xy = env_state.agent_pos[agent_index]
        dir_idx = env_state.agent_dir_idx[agent_index]
        forward, lateral = self._cone_forward_lateral(h, w, pos_xy, dir_idx)

        in_front = forward >= 0
        in_range = forward <= self.fov_range
        in_cone = jnp.abs(lateral) <= (self.fov_slope * forward + 1.0)
        geom_vis = in_front & in_range & in_cone

        if not self.use_occlusion:
            return geom_vis

        return geom_vis & self._occlusion_mask(env_state, agent_index, h, w)

    def _soft_weights(self, env_state: OvercookedState, agent_index: int, h: int, w: int) -> jnp.ndarray:
        """Float weights in [0, 1] inside FOV."""
        pos_xy = env_state.agent_pos[agent_index]
        dir_idx = env_state.agent_dir_idx[agent_index]
        forward, lateral = self._cone_forward_lateral(h, w, pos_xy, dir_idx)

        forward_pos = jnp.maximum(forward, 0)
        w_dist = jnp.exp(-forward_pos / jnp.maximum(self.dist_sigma, 1e-6))
        lat_norm = lateral / (forward_pos + 1.0)
        w_ang = jnp.exp(-(lat_norm * lat_norm) / jnp.maximum(self.ang_sigma * self.ang_sigma, 1e-6))
        return jnp.clip(w_dist * w_ang, 0.0, 1.0)

    def _perception_filter(self, obs: Dict[str, jnp.ndarray], env_state: OvercookedState) -> Dict[str, jnp.ndarray]:
        """Apply FOV mask and optional soft weighting per agent."""
        h, w, _ = obs[self.agents[0]].shape
        filtered_obs = {}

        for agent_idx, agent in enumerate(self.agents):
            mask = self._fov_mask(env_state, agent_idx, h, w)
            if self.soft_view and self.po_mode != "none":
                weights = self._soft_weights(env_state, agent_idx, h, w) * mask.astype(jnp.float32)
                filtered_obs[agent] = obs[agent].astype(jnp.float32) * weights[..., None]
            else:
                filtered_obs[agent] = obs[agent] * mask[..., None].astype(obs[agent].dtype)

        return filtered_obs
    
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        obs, env_state = self.env.reset(key)
        obs = self._perception_filter(obs, env_state)
        flat_obs = {agent: obs[agent].flatten() for agent in self.agents} # flatten obs
        return flat_obs, WrappedEnvState(env_state, jnp.zeros(self.num_agents), jnp.zeros(self.num_agents), jnp.empty((), dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        num_actions = len(self.env.action_set)
        return {agent: jnp.ones(num_actions) for agent in self.agents}
    
    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        """Returns the step count for the environment."""
        return state.env_state.time

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        '''Wrapped step function. The base return is 
        tracked in the info dictionary, so that the return can be obtained from the final info.
        '''
        obs, env_state, rewards, dones, infos = self.env.step(key, state.env_state, actions, reset_state)
        obs = self._perception_filter(obs, env_state)
        flat_obs = {agent: obs[agent].flatten() for agent in self.agents} # flatten obs
        # log the base return in the info
        base_reward = infos['base_reward']
        base_return_so_far = base_reward + state.base_return_so_far
        new_info = {**infos, 'base_return': base_return_so_far}
        
        # handle auto-resetting the base return upon episode termination
        base_return_so_far = jax.lax.select(dones['__all__'], jnp.zeros(self.num_agents), base_return_so_far)
        new_state = WrappedEnvState(env_state=env_state, base_return_so_far=base_return_so_far, avail_actions=jnp.zeros(self.num_agents), step=jnp.empty((), dtype=jnp.int32))
        return flat_obs, new_state, rewards, dones, new_info
