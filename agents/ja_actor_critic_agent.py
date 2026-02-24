"""Policy wrapper for the Joint Attention Actor-Critic.

Mirrors Lee et al. (2021). Key differences from standard RNNActorCriticPolicy:
- get_action_value_policy takes partner_hstate and returns attention maps
- Cross-agent state sharing: partner's LSTM hidden state is passed to the network
- LSTM carry is ((actor_h, actor_c), (critic_h, critic_c))
"""
from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.ja_actor_critic import JAActorCritic, JAScannedLSTM


class JAActorCriticPolicy(AgentPolicy):
    """Policy wrapper for the Joint Attention Actor-Critic.

    Hidden state layout:
      hstate shape: (1, batch, lstm_hidden_dim * 4)
      This packs (actor_h, actor_c, critic_h, critic_c) into a single array
      for compatibility with the existing AgentPolicy interface, which expects
      a single hstate array. Internally we unpack/repack as needed.
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        obs_height: int,
        obs_width: int,
        obs_channels: int = 26,
        activation: str = "relu",
        conv_filters: int = 64,
        num_heads: int = 4,
        head_features: int = 16,
        fc_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        spatial_basis_depth: int = 8,
        scalar_embed_dim: int = 5,
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
            lstm_hidden_dim=lstm_hidden_dim,
            spatial_basis_depth=spatial_basis_depth,
            scalar_embed_dim=scalar_embed_dim,
            activation=activation,
        )
        self.lstm_hidden_dim = lstm_hidden_dim

    def _pack_hstate(self, actor_lstm_state, critic_lstm_state):
        """Pack ((actor_h, actor_c), (critic_h, critic_c)) -> (1, batch, 4*dim)."""
        actor_h, actor_c = actor_lstm_state
        critic_h, critic_c = critic_lstm_state
        # Each is (batch, dim); concatenate along last axis
        packed = jnp.concatenate([actor_h, actor_c, critic_h, critic_c], axis=-1)
        return packed[None, ...]  # (1, batch, 4*dim)

    def _unpack_hstate(self, hstate):
        """Unpack (1, batch, 4*dim) -> ((actor_h, actor_c), (critic_h, critic_c))."""
        flat = hstate.squeeze(0)  # (batch, 4*dim)
        d = self.lstm_hidden_dim
        actor_h = flat[..., 0:d]
        actor_c = flat[..., d:2*d]
        critic_h = flat[..., 2*d:3*d]
        critic_c = flat[..., 3*d:4*d]
        return (actor_h, actor_c), (critic_h, critic_c)

    def _extract_actor_h(self, hstate):
        """Extract just the actor hidden state h from packed hstate.

        This is what gets shared with the partner as partner_hstate.
        hstate: (1, batch, 4*dim) -> actor_h: (1, batch, dim)
        """
        d = self.lstm_hidden_dim
        return hstate[..., 0:d]  # (1, batch, dim)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   partner_hstate=None, aux_obs=None, env_state=None, test_mode=False):
        """Get actions for the JA policy.

        Args:
            hstate: (1, batch, 4*lstm_hidden_dim) packed LSTM states
            partner_hstate: (1, batch, lstm_hidden_dim) partner's actor h.
                If None, defaults to zeros.
        """
        batch_size = obs.shape[1]
        if partner_hstate is None:
            partner_hstate = jnp.zeros((1, batch_size, self.lstm_hidden_dim))

        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, _, _ = self.network.apply(
            params, hidden, (obs, done, avail_actions, partner_hstate)
        )
        action = jax.lax.cond(
            test_mode,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        new_hstate = self._pack_hstate(*new_hidden)
        return action, new_hstate

    @partial(jax.jit, static_argnums=(0,))
    def get_action_and_attention(self, params, obs, done, avail_actions, hstate, rng,
                                 partner_hstate=None, test_mode=False):
        """Like get_action, but also returns the attention map.

        Returns:
            (action, new_hstate, attn_map)
            attn_map shape: (1, batch, H, W)
        """
        batch_size = obs.shape[1]
        if partner_hstate is None:
            partner_hstate = jnp.zeros((1, batch_size, self.lstm_hidden_dim))

        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, _, attn_map = self.network.apply(
            params, hidden, (obs, done, avail_actions, partner_hstate)
        )
        action = jax.lax.cond(
            test_mode,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        new_hstate = self._pack_hstate(*new_hidden)
        return action, new_hstate, attn_map

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                partner_hstate=None, aux_obs=None, env_state=None):
        """Get actions, values, policy, and attention map.

        Args:
            hstate: (1, batch, 4*lstm_hidden_dim) packed LSTM states
            partner_hstate: (seq_len, batch, lstm_hidden_dim) or
                            (1, batch, lstm_hidden_dim). Partner's actor h.
                            If None, defaults to zeros.

        Returns:
            (action, value, pi, new_hstate, attn_map)
            attn_map shape: (seq_len, batch, H, W)
        """
        batch_size = obs.shape[1]
        seq_len = obs.shape[0]
        if partner_hstate is None:
            partner_hstate = jnp.zeros((seq_len, batch_size, self.lstm_hidden_dim))

        # Broadcast partner_hstate to match sequence length if needed
        if partner_hstate.shape[0] != seq_len:
            partner_hstate = jnp.broadcast_to(
                partner_hstate, (seq_len, batch_size, self.lstm_hidden_dim)
            )

        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, val, attn_map = self.network.apply(
            params, hidden, (obs, done, avail_actions, partner_hstate)
        )
        action = pi.sample(seed=rng)
        new_hstate = self._pack_hstate(*new_hidden)
        return action, val, pi, new_hstate, attn_map

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize packed hidden state: (1, batch, 4*lstm_hidden_dim)."""
        d = self.lstm_hidden_dim
        return jnp.zeros((1, batch_size, 4 * d))

    def init_params(self, rng):
        """Initialize parameters for the JA policy."""
        batch_size = 1
        init_hstate = self.init_hstate(batch_size)
        hidden = self._unpack_hstate(init_hstate)

        seq_len = 1
        dummy_obs = jnp.zeros((seq_len, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((seq_len, batch_size))
        dummy_avail = jnp.ones((seq_len, batch_size, self.action_dim))
        dummy_partner_h = jnp.zeros((seq_len, batch_size, self.lstm_hidden_dim))
        dummy_x = (dummy_obs, dummy_done, dummy_avail, dummy_partner_h)

        return self.network.init(rng, hidden, dummy_x)
