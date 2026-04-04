"""Policy wrapper for the gaze-based image actor-critic."""
from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.gaze_image_actor_critic import (
    GazeImageActorCritic,
    VISION_LSTM_CHANNELS,
    _compute_mott_output_dims,
)


class GazeImageActorCriticPolicy(AgentPolicy):
    """Shared-controller policy wrapper around GazeImageActorCritic."""

    uses_prev_reward_action = True

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        img_height: int,
        img_width: int,
        conv_filters: int = 32,
        conv_num_blocks: int = 4,
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        conv_padding: str = "SAME",
        fc_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        gaze_spatial_basis_depth: int = 8,
        gaze_key_dim: int = 16,
        gaze_value_dim: int = 120,
        gaze_num_queries: int = 4,
        query_hidden_dim_1: int = 128,
        query_hidden_dim_2: int = 288,
        num_channels: int = 3,
    ):
        super().__init__(action_dim, obs_dim)
        self.img_height = img_height
        self.img_width = img_width
        self.lstm_hidden_dim = lstm_hidden_dim
        self.feat_h, self.feat_w = _compute_mott_output_dims(img_height, img_width)
        self.vision_state_dim = self.feat_h * self.feat_w * VISION_LSTM_CHANNELS
        self.network = GazeImageActorCritic(
            action_dim=action_dim,
            img_height=img_height,
            img_width=img_width,
            fc_hidden_dim=fc_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            gaze_spatial_basis_depth=gaze_spatial_basis_depth,
            gaze_key_dim=gaze_key_dim,
            gaze_value_dim=gaze_value_dim,
            gaze_num_queries=gaze_num_queries,
            query_hidden_dim_1=query_hidden_dim_1,
            query_hidden_dim_2=query_hidden_dim_2,
            num_channels=num_channels,
        )

    def _pack_hstate(self, lstm_state):
        vis_h, vis_c, pol_h, pol_c = lstm_state
        packed = jnp.concatenate(
            [
                vis_h.reshape(vis_h.shape[0], -1),
                vis_c.reshape(vis_c.shape[0], -1),
                pol_h,
                pol_c,
            ],
            axis=-1,
        )
        return packed[None, ...]

    def _unpack_hstate(self, hstate):
        flat = hstate.squeeze(0)
        v = self.vision_state_dim
        d = self.lstm_hidden_dim
        vis_h = flat[..., 0:v].reshape(-1, self.feat_h, self.feat_w, VISION_LSTM_CHANNELS)
        vis_c = flat[..., v:2 * v].reshape(-1, self.feat_h, self.feat_w, VISION_LSTM_CHANNELS)
        pol_h = flat[..., 2 * v:2 * v + d]
        pol_c = flat[..., 2 * v + d:2 * v + 2 * d]
        return (vis_h, vis_c, pol_h, pol_c)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, greedy=False,
                   prev_reward=None, prev_action=None):
        hidden = self._unpack_hstate(hstate)
        if prev_reward is None:
            prev_reward = jnp.zeros(done.shape, dtype=obs.dtype)
        if prev_action is None:
            prev_action = jnp.zeros(done.shape, dtype=obs.dtype)
        new_hidden, pi, _, _ = self.network.apply(
            params, hidden, (obs, done, avail_actions, prev_reward, prev_action)
        )
        action = jax.lax.cond(
            greedy,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        new_hstate = self._pack_hstate(new_hidden)
        return action, new_hstate

    @partial(jax.jit, static_argnums=(0,))
    def get_action_and_attention(self, params, obs, done, avail_actions, hstate, rng,
                                 greedy=False, agent_id=None,
                                 prev_reward=None, prev_action=None):
        hidden = self._unpack_hstate(hstate)
        if prev_reward is None:
            prev_reward = jnp.zeros(done.shape, dtype=obs.dtype)
        if prev_action is None:
            prev_action = jnp.zeros(done.shape, dtype=obs.dtype)
        new_hidden, pi, _, aux = self.network.apply(
            params, hidden, (obs, done, avail_actions, prev_reward, prev_action)
        )
        action = jax.lax.cond(
            greedy,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        new_hstate = self._pack_hstate(new_hidden)
        return action, new_hstate, aux["attn_map"]

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None,
                                prev_reward=None, prev_action=None):
        hidden = self._unpack_hstate(hstate)
        if prev_reward is None:
            prev_reward = jnp.zeros(done.shape, dtype=obs.dtype)
        if prev_action is None:
            prev_action = jnp.zeros(done.shape, dtype=obs.dtype)
        new_hidden, pi, value, aux = self.network.apply(
            params, hidden, (obs, done, avail_actions, prev_reward, prev_action)
        )
        action = pi.sample(seed=rng)
        new_hstate = self._pack_hstate(new_hidden)
        return action, value, pi, new_hstate, aux

    def init_hstate(self, batch_size, aux_info=None):
        v = self.vision_state_dim
        d = self.lstm_hidden_dim
        return jnp.zeros((1, batch_size, 2 * v + 2 * d))

    def init_params(self, rng):
        batch_size = 1
        hidden = self._unpack_hstate(self.init_hstate(batch_size))
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_avail = jnp.ones((1, batch_size, self.action_dim))
        dummy_prev_reward = jnp.zeros((1, batch_size))
        dummy_prev_action = jnp.zeros((1, batch_size))
        return self.network.init(
            rng,
            hidden,
            (dummy_obs, dummy_done, dummy_avail, dummy_prev_reward, dummy_prev_action),
        )
