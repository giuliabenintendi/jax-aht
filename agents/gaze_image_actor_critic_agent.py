"""Policy wrapper for the gaze-based image actor-critic."""
from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.gaze_image_actor_critic import GazeImageActorCritic


class GazeImageActorCriticPolicy(AgentPolicy):
    """Packed actor/critic-LSTM policy wrapper around GazeImageActorCritic."""

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
        gaze_hidden_dim: int = 64,
        gaze_spatial_basis_depth: int = 8,
        gaze_conv_dim: int = 16,
        contrastive_dim: int = 128,
        num_channels: int = 3,
        target_sigma_x: float = 0.20,
        target_sigma_y: float = 0.20,
    ):
        super().__init__(action_dim, obs_dim)
        self.img_height = img_height
        self.img_width = img_width
        self.lstm_hidden_dim = lstm_hidden_dim
        self.network = GazeImageActorCritic(
            action_dim=action_dim,
            img_height=img_height,
            img_width=img_width,
            conv_filters=conv_filters,
            conv_num_blocks=conv_num_blocks,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
            fc_hidden_dim=fc_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            gaze_hidden_dim=gaze_hidden_dim,
            gaze_spatial_basis_depth=gaze_spatial_basis_depth,
            gaze_conv_dim=gaze_conv_dim,
            contrastive_dim=contrastive_dim,
            num_channels=num_channels,
            target_sigma_x=target_sigma_x,
            target_sigma_y=target_sigma_y,
        )

    def _pack_hstate(self, actor_lstm_state, critic_lstm_state):
        actor_h, actor_c = actor_lstm_state
        critic_h, critic_c = critic_lstm_state
        packed = jnp.concatenate([actor_h, actor_c, critic_h, critic_c], axis=-1)
        return packed[None, ...]

    def _unpack_hstate(self, hstate):
        flat = hstate.squeeze(0)
        d = self.lstm_hidden_dim
        actor_h = flat[..., 0:d]
        actor_c = flat[..., d:2 * d]
        critic_h = flat[..., 2 * d:3 * d]
        critic_c = flat[..., 3 * d:4 * d]
        return (actor_h, actor_c), (critic_h, critic_c)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, greedy=False):
        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, _, _ = self.network.apply(
            params, hidden, (obs, done, avail_actions)
        )
        action = jax.lax.cond(
            greedy,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        new_hstate = self._pack_hstate(*new_hidden)
        return action, new_hstate

    @partial(jax.jit, static_argnums=(0,))
    def get_action_and_attention(self, params, obs, done, avail_actions, hstate, rng,
                                 greedy=False, agent_id=None):
        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, _, aux = self.network.apply(
            params, hidden, (obs, done, avail_actions)
        )
        action = jax.lax.cond(
            greedy,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        new_hstate = self._pack_hstate(*new_hidden)
        return action, new_hstate, aux["attn_map"]

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, value, aux = self.network.apply(
            params, hidden, (obs, done, avail_actions)
        )
        action = pi.sample(seed=rng)
        new_hstate = self._pack_hstate(*new_hidden)
        return action, value, pi, new_hstate, aux

    @partial(jax.jit, static_argnums=(0,))
    def get_contrastive_features(self, params, obs):
        return self.network.apply(
            params, obs, method=self.network.contrastive_features
        )

    def init_hstate(self, batch_size, aux_info=None):
        d = self.lstm_hidden_dim
        return jnp.zeros((1, batch_size, 4 * d))

    def init_params(self, rng):
        batch_size = 1
        hidden = self._unpack_hstate(self.init_hstate(batch_size))
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_avail = jnp.ones((1, batch_size, self.action_dim))
        return self.network.init(rng, hidden, (dummy_obs, dummy_done, dummy_avail))
