"""Joint Attention policy wrappers."""
from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.ja_actor_critic import JAActorCritic


class JAActorCriticPolicy(AgentPolicy):
    """Packed-LSTM policy wrapper for the Joint Attention actor-critic."""

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        obs_height: int | None = None,
        obs_width: int | None = None,
        obs_channels: int = 26,
        img_height: int | None = None,
        img_width: int | None = None,
        activation: str = "relu",
        encoder_type: str = "symbolic",
        conv_filters: int = 64,
        conv_num_blocks: int = 4,
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        conv_padding: str = "SAME",
        num_heads: int = 4,
        head_features: int = 16,
        fc_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        spatial_basis_depth: int = 8,
        scalar_embed_dim: int = 5,
        num_channels: int = 3,
        message_dim: int = 0,
        scalar_dim: int = 0,
        query_partner_lstm: bool = False,
        post_lstm_fc_layers: int = 0,
        post_lstm_fc_hidden_dim: int | None = None,
    ):
        super().__init__(action_dim, obs_dim)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.query_partner_lstm = query_partner_lstm
        self.network = JAActorCritic(
            action_dim=action_dim,
            encoder_type=encoder_type,
            obs_height=obs_height,
            obs_width=obs_width,
            obs_channels=obs_channels,
            img_height=img_height,
            img_width=img_width,
            num_channels=num_channels,
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
            scalar_embed_dim=scalar_embed_dim,
            message_dim=message_dim,
            scalar_dim=scalar_dim,
            query_partner_lstm=query_partner_lstm,
            post_lstm_fc_layers=post_lstm_fc_layers,
            post_lstm_fc_hidden_dim=post_lstm_fc_hidden_dim,
            activation=activation,
        )

    def _pack_hstate(self, actor_lstm_state, critic_lstm_state):
        actor_h, actor_c = actor_lstm_state
        critic_h, critic_c = critic_lstm_state
        return jnp.concatenate([actor_h, actor_c, critic_h, critic_c], axis=-1)[None, ...]

    def _unpack_hstate(self, hstate):
        flat = hstate.squeeze(0)
        d = self.lstm_hidden_dim
        actor_h = flat[..., 0:d]
        actor_c = flat[..., d : 2 * d]
        critic_h = flat[..., 2 * d : 3 * d]
        critic_c = flat[..., 3 * d : 4 * d]
        return (actor_h, actor_c), (critic_h, critic_c)

    def _build_network_input(self, obs, done, avail_actions, plh_actor=None, plh_critic=None):
        if not self.query_partner_lstm:
            return obs, done, avail_actions

        seq_len, batch_size = obs.shape[:2]
        if plh_actor is None:
            plh_actor = jnp.zeros((seq_len, batch_size, self.lstm_hidden_dim))
        if plh_critic is None:
            plh_critic = jnp.zeros((seq_len, batch_size, self.lstm_hidden_dim))
        return obs, done, avail_actions, plh_actor, plh_critic

    @partial(jax.jit, static_argnums=(0,))
    def get_action(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        aux_obs=None,
        env_state=None,
        greedy=False,
        agent_id=None,
    ):
        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, _, _ = self.network.apply(
            params,
            hidden,
            self._build_network_input(obs, done, avail_actions),
        )
        action = jax.lax.cond(greedy, lambda: pi.mode(), lambda: pi.sample(seed=rng))
        return action, self._pack_hstate(*new_hidden)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_and_attention(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        greedy=False,
        agent_id=None,
        plh_actor=None,
        plh_critic=None,
        prev_reward=None,
        prev_action=None,
    ):
        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, _, attn_map = self.network.apply(
            params,
            hidden,
            self._build_network_input(
                obs,
                done,
                avail_actions,
                plh_actor=plh_actor,
                plh_critic=plh_critic,
            ),
        )
        action = jax.lax.cond(greedy, lambda: pi.mode(), lambda: pi.sample(seed=rng))
        return action, self._pack_hstate(*new_hidden), attn_map

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(
        self,
        params,
        obs,
        done,
        avail_actions,
        hstate,
        rng,
        aux_obs=None,
        env_state=None,
        plh_actor=None,
        plh_critic=None,
    ):
        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, val, attn_map = self.network.apply(
            params,
            hidden,
            self._build_network_input(
                obs,
                done,
                avail_actions,
                plh_actor=plh_actor,
                plh_critic=plh_critic,
            ),
        )
        action = pi.sample(seed=rng)
        return action, val, pi, self._pack_hstate(*new_hidden), attn_map

    def init_hstate(self, batch_size, aux_info=None):
        return jnp.zeros((1, batch_size, 4 * self.lstm_hidden_dim))

    def init_params(self, rng):
        batch_size = 1
        seq_len = 1
        hidden = self._unpack_hstate(self.init_hstate(batch_size))
        dummy_obs = jnp.zeros((seq_len, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((seq_len, batch_size))
        dummy_avail = jnp.ones((seq_len, batch_size, self.action_dim))
        dummy_x = self._build_network_input(dummy_obs, dummy_done, dummy_avail)
        return self.network.init(rng, hidden, dummy_x)


class JAImageActorCriticPolicy(JAActorCriticPolicy):
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
        message_dim: int = 0,
        scalar_dim: int = 0,
        scalar_embed_dim: int = 5,
        query_partner_lstm: bool = False,
    ):
        super().__init__(
            action_dim=action_dim,
            obs_dim=obs_dim,
            img_height=img_height,
            img_width=img_width,
            encoder_type="image",
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
            message_dim=message_dim,
            scalar_dim=scalar_dim,
            scalar_embed_dim=scalar_embed_dim,
            query_partner_lstm=query_partner_lstm,
            post_lstm_fc_layers=2,
        )
