"""Joint Attention Actor-Critic network.

Mirrors the architecture from Lee et al. (2021), Appendix A:
  "Joint Attention for Multi-Agent Coordination and Social Learning"

Per-agent architecture (actor and critic are identical but share no weights):
  obs (flat) -> unflatten (H, W, 26)
  Agent positions (0-1) + terrain/objects (10-25) -> Conv(3x3, 64, SAME, ReLU) -> F
  F + sinusoidal spatial basis (depth 8) -> 1x1 Conv -> Keys K, Values V
  Queries Q = Dense(concat(h_ego, h_partner))  [top-down, cross-agent]
  Multi-head attention (4 heads, depth 16): softmax(Q . K) -> attended O
  Scalar features: direction (2-9) -> Dense(5),
                    ego+partner position (0-1) -> Dense(5)
  LSTM(concat(O, dir_embed, pos_embed), (h, c)) -> h_t
  Dense(64) -> act -> Dense(64) -> act -> output

Actor output: action logits
Critic output: scalar value
Attention map A (mean over heads) returned for the JA incentive.
"""
import functools

import numpy as np
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp

from agents.ja_utils import make_sinusoidal_spatial_basis


# Observation channel layout (Overcooked lossless encoding, 26 channels):
#   0-1:   agent positions (ego, partner)
#   2-5:   ego direction one-hot (N, S, E, W)
#   6-9:   partner direction one-hot (N, S, E, W)
#   10-25: image channels (terrain, pots, objects, urgency)
_POS_CHANNELS = slice(0, 2)
_EGO_DIR_CHANNELS = slice(2, 6)
_PARTNER_DIR_CHANNELS = slice(6, 10)
_IMAGE_CHANNELS = slice(10, 26)


class JAScannedLSTM(nn.Module):
    """Scanned module: spatial attention + LSTM, mirroring Lee et al. (2021).

    At each timestep:
      1. Unflatten obs -> grid (H, W, 26)
      2. Agent positions (0-1) + terrain (10-25) -> Conv(3x3, 64) -> features F
      3. F + sinusoidal spatial basis (depth 8) -> 1x1 Conv -> Keys K, Values V
      4. Q = Dense(concat(h_ego, h_partner)) — cross-agent, top-down query
      5. Multi-head attention (no sqrt scaling): A = softmax(Q . K), O = sum(A * V)
      6. Extract scalars: direction -> Dense(5), ego+partner pos -> Dense(5)
      7. LSTM(concat(O, dir_embed, pos_embed), (h, c)) -> (h_t, c_t)
      8. Return (h_t, c_t) and attention map (averaged over heads)
    """
    obs_height: int
    obs_width: int
    obs_channels: int = 26
    conv_filters: int = 64
    num_heads: int = 4
    head_features: int = 16
    lstm_hidden_dim: int = 64
    spatial_basis_depth: int = 8
    scalar_embed_dim: int = 5

    def setup(self):
        self.spatial_basis = make_sinusoidal_spatial_basis(
            self.obs_height, self.obs_width, self.spatial_basis_depth
        )

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Process one timestep.

        Args:
            carry: LSTM state tuple (h, c), each (batch, lstm_hidden_dim)
            x: (obs, dones, partner_hstate) where
               obs is (batch, obs_flat_dim),
               dones is (batch,),
               partner_hstate is (batch, lstm_hidden_dim) — the partner's h

        Returns:
            new_carry: updated LSTM state (h, c)
            (lstm_h, attn_map): hidden output for FC heads, and
                attention map averaged over heads (batch, H, W)
        """
        lstm_h, lstm_c = carry
        obs_flat, dones, partner_hstate = x

        batch_size = obs_flat.shape[0]
        h, w = self.obs_height, self.obs_width
        m, cm = self.num_heads, self.head_features

        # Reset LSTM state on episode boundaries
        zero_h, zero_c = self.initialize_carry(batch_size, self.lstm_hidden_dim)
        lstm_h = jnp.where(dones[:, np.newaxis], zero_h, lstm_h)
        lstm_c = jnp.where(dones[:, np.newaxis], zero_c, lstm_c)

        # --- 1. Unflatten observation to grid ---
        obs_grid = obs_flat.reshape(batch_size, h, w, self.obs_channels)

        # --- 2. Agent positions + terrain -> Conv -> features ---
        # Channels 0-1 (agent positions) are included so the conv layer sees where
        # both agents are spatially, matching MultiGrid where agents are in the image.
        agent_pos = obs_grid[..., _POS_CHANNELS]    # (batch, H, W, 2)
        terrain = obs_grid[..., _IMAGE_CHANNELS]    # (batch, H, W, 16)
        image = jnp.concatenate([agent_pos, terrain], axis=-1)  # (batch, H, W, 18)
        features = nn.Conv(
            features=self.conv_filters,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="feature_conv",
        )(image)
        features = nn.relu(features)  # (batch, H, W, conv_filters)

        # --- 3. Append spatial basis, compute K and V via 1x1 conv ---
        spatial = jnp.broadcast_to(
            self.spatial_basis[None, ...],
            (batch_size, h, w, self.spatial_basis_depth),
        )
        features_with_pos = jnp.concatenate([features, spatial], axis=-1)

        keys = nn.Conv(
            features=m * cm, kernel_size=(1, 1),
            kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="key_conv",
        )(features_with_pos)
        keys = keys.reshape(batch_size, h * w, m, cm)

        values = nn.Conv(
            features=m * cm, kernel_size=(1, 1),
            kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="value_conv",
        )(features_with_pos)
        values = values.reshape(batch_size, h * w, m, cm)

        # --- 4. Cross-agent query: Q = Dense(concat(h_ego, h_partner)) ---
        # Paper: Q_i = f_Q(c_θ^i, c_θ^j) where c_θ is the recurrent state.
        # We use h (the LSTM output) as the recurrent state for both agents.
        query_input = jnp.concatenate([lstm_h, partner_hstate], axis=-1)
        queries = nn.Dense(
            m * cm, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="query_ffn",
        )(query_input)
        queries = queries.reshape(batch_size, m, cm)

        # --- 5. Multi-head spatial attention ---
        # No sqrt(d) scaling — matches reference code (raw dot products)
        attn_logits = jnp.einsum("bnmc,bmc->bnm", keys, queries)
        attn_weights = jax.nn.softmax(attn_logits, axis=1)  # (batch, HW, m)

        attended = jnp.einsum("bnm,bnmc->bmc", attn_weights, values)
        attended_flat = attended.reshape(batch_size, m * cm)

        # Attention map averaged over heads for JA incentive
        attn_map = attn_weights.mean(axis=-1).reshape(batch_size, h, w)

        # --- 6. Scalar features: direction and position ---
        ego_dir = obs_grid[..., _EGO_DIR_CHANNELS].sum(axis=(1, 2))
        partner_dir = obs_grid[..., _PARTNER_DIR_CHANNELS].sum(axis=(1, 2))
        direction = jnp.concatenate([ego_dir, partner_dir], axis=-1)  # (batch, 8)
        dir_embed = nn.Dense(
            self.scalar_embed_dim,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="dir_embed",
        )(direction)

        # Ego + partner positions as scalar (x, y) coordinates.
        # Mirrors dir_embed which also receives both agents' features concatenated.
        ego_pos_grid = obs_grid[..., 0]  # (batch, H, W)
        ego_flat = jnp.argmax(ego_pos_grid.reshape(batch_size, -1), axis=-1)
        ego_y = ego_flat // w
        ego_x = ego_flat % w
        partner_pos_grid = obs_grid[..., 1]  # (batch, H, W)
        partner_flat = jnp.argmax(partner_pos_grid.reshape(batch_size, -1), axis=-1)
        partner_y = partner_flat // w
        partner_x = partner_flat % w
        pos_features = jnp.stack(
            [ego_x.astype(jnp.float32), ego_y.astype(jnp.float32),
             partner_x.astype(jnp.float32), partner_y.astype(jnp.float32)],
            axis=-1,
        )  # (batch, 4)
        pos_embed = nn.Dense(
            self.scalar_embed_dim,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="pos_embed",
        )(pos_features)

        # --- 7. LSTM update ---
        lstm_input = jnp.concatenate([attended_flat, dir_embed, pos_embed], axis=-1)
        new_carry, lstm_out = nn.OptimizedLSTMCell(
            features=self.lstm_hidden_dim,
        )((lstm_h, lstm_c), lstm_input)
        new_h, new_c = new_carry

        return (new_h, new_c), (lstm_out, attn_map)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """Returns (h, c) tuple, each of shape (batch_size, hidden_size)."""
        return (
            jnp.zeros((batch_size, hidden_size)),
            jnp.zeros((batch_size, hidden_size)),
        )


class JAActorCritic(nn.Module):
    """Joint Attention Actor-Critic, mirroring Lee et al. (2021).

    Actor and critic are identical architectures but share no weights,
    matching the paper: "The value network is identical to the policy
    network and shares no weights."

    Each path: Conv -> Attention -> LSTM -> Dense(64) -> Dense(64) -> output
    Only the actor path's attention map is returned for the JA incentive.
    """
    action_dim: int
    obs_height: int
    obs_width: int
    obs_channels: int = 26
    conv_filters: int = 64
    num_heads: int = 4
    head_features: int = 16
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
    spatial_basis_depth: int = 8
    scalar_embed_dim: int = 5
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        """Forward pass through both actor and critic networks.

        Args:
            hidden: tuple of (actor_lstm_state, critic_lstm_state)
                each lstm_state is ((h, c), ) where h, c are (batch, lstm_dim)
            x: (obs, dones, avail_actions, partner_hstate)

        Returns:
            new_hidden, pi, value, attn_map
        """
        activation = nn.relu if self.activation == "relu" else nn.tanh
        obs, dones, avail_actions, partner_hstate = x

        actor_lstm_state, critic_lstm_state = hidden

        rnn_kwargs = dict(
            obs_height=self.obs_height,
            obs_width=self.obs_width,
            obs_channels=self.obs_channels,
            conv_filters=self.conv_filters,
            num_heads=self.num_heads,
            head_features=self.head_features,
            lstm_hidden_dim=self.lstm_hidden_dim,
            spatial_basis_depth=self.spatial_basis_depth,
            scalar_embed_dim=self.scalar_embed_dim,
        )

        # --- Actor path (own conv, attention, LSTM) ---
        actor_lstm_state, (actor_embed, attn_map) = JAScannedLSTM(
            **rnn_kwargs, name="actor_lstm",
        )(actor_lstm_state, (obs, dones, partner_hstate))

        actor_out = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0),
            name="actor_fc1",
        )(actor_embed)
        actor_out = activation(actor_out)
        actor_out = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0),
            name="actor_fc2",
        )(actor_out)
        actor_out = activation(actor_out)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
            name="actor_proj",
        )(actor_out)

        unavail_actions = 1 - avail_actions
        action_logits = action_logits - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        # --- Critic path (own conv, attention, LSTM — no shared weights) ---
        critic_lstm_state, (critic_embed, _) = JAScannedLSTM(
            **rnn_kwargs, name="critic_lstm",
        )(critic_lstm_state, (obs, dones, partner_hstate))

        critic_out = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0),
            name="critic_fc1",
        )(critic_embed)
        critic_out = activation(critic_out)
        critic_out = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0),
            name="critic_fc2",
        )(critic_out)
        critic_out = activation(critic_out)
        value = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="critic_proj",
        )(critic_out)

        new_hidden = (actor_lstm_state, critic_lstm_state)
        return new_hidden, pi, jnp.squeeze(value, axis=-1), attn_map
