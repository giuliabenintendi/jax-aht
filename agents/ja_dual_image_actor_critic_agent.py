"""Policy wrapper for the Joint Attention Dual-Critic with image observations.

Same hstate layout as JAActorCriticPolicy (actor_h, actor_c, critic_h, critic_c).
The only difference is get_action_value_policy returns (val_ext, val_int) instead
of a single value.
"""
from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.ja_dual_image_actor_critic import JADualImageActorCritic
from agents.ja_image_actor_critic import _compute_resnet_output_dims


class JADualImageActorCriticPolicy(AgentPolicy):
    """Policy wrapper for dual-critic JA network with image observations.

    Hidden state layout identical to single-critic version:
      hstate shape: (1, batch, lstm_hidden_dim * 4)
    """

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
        num_heads: int = 4,
        head_features: int = 16,
        fc_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        spatial_basis_depth: int = 8,
        num_channels: int = 3,
        feed_other_attn_mode: str = "channel",
        partner_attn_gain: float = 1.0,
    ):
        super().__init__(action_dim, obs_dim)
        self.img_height = img_height
        self.img_width = img_width
        self.feed_other_attn_mode = feed_other_attn_mode
        self.feat_h, self.feat_w = _compute_resnet_output_dims(
            img_height, img_width, conv_stride, conv_kernel_size, conv_padding, conv_num_blocks,
        )
        self.network = JADualImageActorCritic(
            action_dim=action_dim,
            img_height=img_height,
            img_width=img_width,
            conv_filters=conv_filters,
            conv_num_blocks=conv_num_blocks,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
            num_heads=num_heads,
            head_features=head_features,
            fc_hidden_dim=fc_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            spatial_basis_depth=spatial_basis_depth,
            num_channels=num_channels,
            feed_other_attn_mode=feed_other_attn_mode,
            partner_attn_gain=partner_attn_gain,
        )
        self.lstm_hidden_dim = lstm_hidden_dim

    def _format_other_attn(self, aux_obs, obs):
        if aux_obs is not None:
            return aux_obs
        seq_len, batch_size = obs.shape[:2]
        return jnp.zeros((seq_len, batch_size, self.feat_h, self.feat_w), dtype=obs.dtype)

    def _pack_hstate(self, actor_lstm_state, critic_lstm_state):
        """Pack ((actor_h, actor_c), (critic_h, critic_c)) -> (1, batch, 4*dim)."""
        actor_h, actor_c = actor_lstm_state
        critic_h, critic_c = critic_lstm_state
        packed = jnp.concatenate([actor_h, actor_c, critic_h, critic_c], axis=-1)
        return packed[None, ...]

    def _unpack_hstate(self, hstate):
        """Unpack (1, batch, 4*dim) -> ((actor_h, actor_c), (critic_h, critic_c))."""
        flat = hstate.squeeze(0)
        d = self.lstm_hidden_dim
        actor_h = flat[..., 0:d]
        actor_c = flat[..., d:2*d]
        critic_h = flat[..., 2*d:3*d]
        critic_c = flat[..., 3*d:4*d]
        return (actor_h, actor_c), (critic_h, critic_c)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, greedy=False, agent_id=None):
        hidden = self._unpack_hstate(hstate)
        other_attn = self._format_other_attn(aux_obs, obs)
        new_hidden, pi, _, _, _ = self.network.apply(
            params, hidden, (obs, done, avail_actions, other_attn)
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
                                 greedy=False, agent_id=None, aux_obs=None):
        hidden = self._unpack_hstate(hstate)
        other_attn = self._format_other_attn(aux_obs, obs)
        new_hidden, pi, _, _, attn_map = self.network.apply(
            params, hidden, (obs, done, avail_actions, other_attn)
        )
        action = jax.lax.cond(
            greedy,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        new_hstate = self._pack_hstate(*new_hidden)
        return action, new_hstate, attn_map

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Returns (action, (val_ext, val_int), pi, new_hstate, attn_map)."""
        hidden = self._unpack_hstate(hstate)
        other_attn = self._format_other_attn(aux_obs, obs)
        new_hidden, pi, val_ext, val_int, attn_map = self.network.apply(
            params, hidden, (obs, done, avail_actions, other_attn)
        )
        action = pi.sample(seed=rng)
        new_hstate = self._pack_hstate(*new_hidden)
        return action, (val_ext, val_int), pi, new_hstate, attn_map

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize packed hidden state: (1, batch, 4*lstm_hidden_dim)."""
        d = self.lstm_hidden_dim
        return jnp.zeros((1, batch_size, 4 * d))

    def init_params(self, rng):
        batch_size = 1
        init_hstate = self.init_hstate(batch_size)
        hidden = self._unpack_hstate(init_hstate)
        seq_len = 1
        dummy_obs = jnp.zeros((seq_len, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((seq_len, batch_size))
        dummy_avail = jnp.ones((seq_len, batch_size, self.action_dim))
        dummy_other_attn = jnp.zeros((seq_len, batch_size, self.feat_h, self.feat_w))
        dummy_x = (dummy_obs, dummy_done, dummy_avail, dummy_other_attn)
        return self.network.init(rng, hidden, dummy_x)
