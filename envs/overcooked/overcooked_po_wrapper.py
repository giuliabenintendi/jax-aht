"""Partial observability wrapper for Overcooked-v1.

Extends the base OvercookedWrapper with cone-shaped FOV masking,
optional occlusion, and soft distance/angular weighting.
"""
from typing import Dict

import jax.numpy as jnp
from jaxmarl.environments.overcooked.overcooked import State as OvercookedState

from envs.overcooked.overcooked_wrapper import OvercookedWrapper
from envs.overcooked.po_utils import cone_forward_lateral


class OvercookedPOWrapper(OvercookedWrapper):
    '''OvercookedWrapper with partial observability via forward cone FOV.

    Observations outside the agent's field of view are zeroed out.
    When soft_view is enabled, visible cells are weighted by distance
    and angular proximity to the agent's facing direction.
    '''
    def __init__(
        self,
        *args,
        po_mode: str = "cone",
        fov_range: int | None = None,
        fov_slope: float = 0.7,
        use_occlusion: bool = False,
        soft_view: bool = True,
        dist_sigma: float = 3.0,
        ang_sigma: float = 1.5,
        **kwargs,
    ):
        if po_mode not in {"none", "cone"}:
            raise ValueError(f"Unsupported po_mode '{po_mode}'. Expected one of: 'none', 'cone'.")

        super().__init__(*args, **kwargs)

        self.po_mode = po_mode
        if fov_range is None:
            h, w = self.env.obs_shape[1], self.env.obs_shape[0]
            fov_range = max(h, w) // 2
        self.fov_range = fov_range
        self.fov_slope = fov_slope
        self.use_occlusion = use_occlusion
        self.soft_view = soft_view
        self.dist_sigma = dist_sigma
        self.ang_sigma = ang_sigma

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
        forward, lateral = cone_forward_lateral(h, w, pos_xy, dir_idx)

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
        forward, lateral = cone_forward_lateral(h, w, pos_xy, dir_idx)

        forward_pos = jnp.maximum(forward, 0)
        w_dist = jnp.exp(-forward_pos / jnp.maximum(self.dist_sigma, 1e-6))
        lat_norm = lateral / (forward_pos + 1.0)
        w_ang = jnp.exp(-(lat_norm * lat_norm) / jnp.maximum(self.ang_sigma * self.ang_sigma, 1e-6))
        return jnp.clip(w_dist * w_ang, 0.0, 1.0)

    def _filter_obs(self, obs: Dict[str, jnp.ndarray], env_state: OvercookedState) -> Dict[str, jnp.ndarray]:
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
