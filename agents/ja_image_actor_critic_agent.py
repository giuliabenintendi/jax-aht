"""Policy wrapper for the Joint Attention Actor-Critic with image observations.

Inherits all hstate packing/unpacking and action methods from JAActorCriticPolicy.
Only overrides __init__ to construct JAImageActorCritic instead of JAActorCritic.
"""
from functools import partial

import jax
import jax.numpy as jnp

from agents.ja_actor_critic_agent import JAActorCriticPolicy
from agents.ja_image_actor_critic import JAImageActorCritic, _compute_resnet_output_dims


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
        cross_agent_attn: bool = False,
        query_partner_lstm: bool = False,
    ):
        # Skip JAActorCriticPolicy.__init__ — we set self.network directly
        # but still call AgentPolicy.__init__ for action_dim/obs_dim
        from agents.agent_interface import AgentPolicy
        AgentPolicy.__init__(self, action_dim, obs_dim)

        self.img_height = img_height
        self.img_width = img_width
        self.message_dim = message_dim
        self.cross_agent_attn = cross_agent_attn
        self.query_partner_lstm = query_partner_lstm
        # Cross-attention now exchanges spatial features (H*W, feat_dim)
        feat_h, feat_w = _compute_resnet_output_dims(
            img_height, img_width, conv_stride, conv_kernel_size, conv_padding, conv_num_blocks)
        self.xattn_num_positions = feat_h * feat_w
        self.xattn_feat_dim = conv_filters + spatial_basis_depth
        self.network = JAImageActorCritic(
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
            message_dim=message_dim,
            scalar_dim=scalar_dim,
            scalar_embed_dim=scalar_embed_dim,
            cross_agent_attn=cross_agent_attn,
            query_partner_lstm=query_partner_lstm,
        )
        self.lstm_hidden_dim = lstm_hidden_dim

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None,
                                partner_embed_actor=None, partner_embed_critic=None,
                                plh_actor=None, plh_critic=None):
        hidden = self._unpack_hstate(hstate)
        if self.query_partner_lstm:
            if plh_actor is None:
                plh_actor = jnp.zeros((*obs.shape[:2], self.lstm_hidden_dim))
            if plh_critic is None:
                plh_critic = jnp.zeros((*obs.shape[:2], self.lstm_hidden_dim))

        # Build input tuple
        x_parts = [obs, done, avail_actions]
        if self.cross_agent_attn:
            if partner_embed_actor is None:
                partner_embed_actor = jnp.zeros((*obs.shape[:2], self.xattn_num_positions, self.xattn_feat_dim))
            if partner_embed_critic is None:
                partner_embed_critic = jnp.zeros((*obs.shape[:2], self.xattn_num_positions, self.xattn_feat_dim))
            x_parts.extend([partner_embed_actor, partner_embed_critic])
        if self.query_partner_lstm:
            x_parts.extend([plh_actor, plh_critic])
        x = tuple(x_parts)

        if self.cross_agent_attn:
            new_hidden, pi, val, attn_map, actor_own_embed, critic_own_embed = \
                self.network.apply(params, hidden, x)
            action = pi.sample(seed=rng)
            new_hstate = self._pack_hstate(*new_hidden)
            return action, val, pi, new_hstate, attn_map, actor_own_embed, critic_own_embed
        else:
            new_hidden, pi, val, attn_map = self.network.apply(params, hidden, x)
            action = pi.sample(seed=rng)
            new_hstate = self._pack_hstate(*new_hidden)
            return action, val, pi, new_hstate, attn_map

    @partial(jax.jit, static_argnums=(0,))
    def get_action_and_attention(self, params, obs, done, avail_actions, hstate, rng,
                                 greedy=False, agent_id=None,
                                 partner_embed_actor=None, partner_embed_critic=None,
                                 plh_actor=None, plh_critic=None,
                                 prev_reward=None, prev_action=None):
        hidden = self._unpack_hstate(hstate)
        if self.query_partner_lstm:
            if plh_actor is None:
                plh_actor = jnp.zeros((*obs.shape[:2], self.lstm_hidden_dim))
            if plh_critic is None:
                plh_critic = jnp.zeros((*obs.shape[:2], self.lstm_hidden_dim))

        # Build input tuple
        needs_custom_path = self.cross_agent_attn or self.query_partner_lstm
        if needs_custom_path:
            x_parts = [obs, done, avail_actions]
            if self.cross_agent_attn:
                if partner_embed_actor is None:
                    partner_embed_actor = jnp.zeros((*obs.shape[:2], self.xattn_num_positions, self.xattn_feat_dim))
                if partner_embed_critic is None:
                    partner_embed_critic = jnp.zeros((*obs.shape[:2], self.xattn_num_positions, self.xattn_feat_dim))
                x_parts.extend([partner_embed_actor, partner_embed_critic])
            if self.query_partner_lstm:
                x_parts.extend([plh_actor, plh_critic])
            x = tuple(x_parts)

            if self.cross_agent_attn:
                new_hidden, pi, _, attn_map, actor_own_embed, critic_own_embed = \
                    self.network.apply(params, hidden, x)
                action = jax.lax.cond(
                    greedy,
                    lambda: pi.mode(),
                    lambda: pi.sample(seed=rng),
                )
                new_hstate = self._pack_hstate(*new_hidden)
                return action, new_hstate, attn_map, actor_own_embed, critic_own_embed
            else:
                new_hidden, pi, _, attn_map = self.network.apply(params, hidden, x)
                action = jax.lax.cond(
                    greedy,
                    lambda: pi.mode(),
                    lambda: pi.sample(seed=rng),
                )
                new_hstate = self._pack_hstate(*new_hidden)
                return action, new_hstate, attn_map
        else:
            # Fall back to base class behavior (3 return values)
            return super().get_action_and_attention(
                params, obs, done, avail_actions, hstate, rng,
                greedy=greedy, agent_id=agent_id,
                prev_reward=prev_reward, prev_action=prev_action,
            )

    def init_params(self, rng):
        batch_size = 1
        init_hstate = self.init_hstate(batch_size)
        hidden = self._unpack_hstate(init_hstate)
        seq_len = 1
        dummy_obs = jnp.zeros((seq_len, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((seq_len, batch_size))
        dummy_avail = jnp.ones((seq_len, batch_size, self.action_dim))
        x_parts = [dummy_obs, dummy_done, dummy_avail]
        if self.cross_agent_attn:
            dummy_pe = jnp.zeros((seq_len, batch_size, self.xattn_num_positions, self.xattn_feat_dim))
            x_parts.extend([dummy_pe, dummy_pe])
        if self.query_partner_lstm:
            dummy_plh = jnp.zeros((seq_len, batch_size, self.lstm_hidden_dim))
            x_parts.extend([dummy_plh, dummy_plh])
        dummy_x = tuple(x_parts)
        return self.network.init(rng, hidden, dummy_x)
