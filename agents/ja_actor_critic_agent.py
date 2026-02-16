"""Policy wrapper for the Joint Attention Actor-Critic.

Implements the AgentPolicy interface. The key difference from RNNActorCriticPolicy
is that get_action_value_policy returns an additional attention map output,
needed for computing the JA intrinsic reward.
"""
from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.ja_actor_critic import JAActorCritic, JAScannedRNN


class JAActorCriticPolicy(AgentPolicy):
    """Policy wrapper for the Joint Attention Actor-Critic."""

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        obs_height: int,
        obs_width: int,
        obs_channels: int = 26,
        activation: str = "tanh",
        conv_filters: int = 32,
        num_heads: int = 4,
        head_features: int = 16,
        fc_hidden_dim: int = 64,
        gru_hidden_dim: int = 64,
    ):
        super().__init__(action_dim, obs_dim)
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.network = JAActorCritic(
            action_dim=action_dim,
            obs_height=obs_height,
            obs_width=obs_width,
            obs_channels=obs_channels,
            conv_filters=conv_filters,
            num_heads=num_heads,
            head_features=head_features,
            fc_hidden_dim=fc_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            activation=activation,
        )
        self.gru_hidden_dim = gru_hidden_dim

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the JA policy.

        Shape conventions match RNNActorCriticPolicy:
          obs, done, avail_actions: (seq_len, batch_size, ...)
          hstate: (1, batch_size, gru_hidden_dim)
        """
        batch_size = obs.shape[1]
        new_hstate, pi, _, _ = self.network.apply(
            params, hstate.squeeze(0), (obs, done, avail_actions)
        )
        action = jax.lax.cond(
            test_mode,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        return action, new_hstate.reshape(1, batch_size, -1)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Get actions, values, policy, and attention map.

        Returns:
            (action, value, pi, new_hstate, attn_map)
            attn_map has shape (seq_len, batch_size, H, W) — averaged over heads,
            representing the ego agent's learned attention distribution.
        """
        batch_size = obs.shape[1]
        new_hstate, pi, val, attn_map = self.network.apply(
            params, hstate.squeeze(0), (obs, done, avail_actions)
        )
        action = pi.sample(seed=rng)
        return action, val, pi, new_hstate.reshape(1, batch_size, -1), attn_map

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize hidden state for the JA policy."""
        hstate = JAScannedRNN.initialize_carry(batch_size, self.gru_hidden_dim)
        hstate = hstate.reshape(1, batch_size, self.gru_hidden_dim)
        return hstate

    def init_params(self, rng):
        """Initialize parameters for the JA policy."""
        batch_size = 1
        init_hstate = self.init_hstate(batch_size)

        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_avail = jnp.ones((1, batch_size, self.action_dim))
        dummy_x = (dummy_obs, dummy_done, dummy_avail)

        return self.network.init(rng, init_hstate.reshape(batch_size, -1), dummy_x)
