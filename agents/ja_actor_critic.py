"""Joint Attention Actor-Critic network.

Implements the recurrent visual attention architecture from:
  Lee et al., "Joint Attention for Multi-Agent Coordination and Social Learning", 2021.

Architecture:
  obs (flat) -> unflatten (H, W, 26) -> Conv -> features F
  F + spatial basis S -> Conv -> Keys K, Values V
  Queries Q from GRU hidden state (top-down, goal-directed)
  Multi-head spatial attention: softmax(Q . K) over (H, W) -> attended output O
  O -> GRU -> actor / critic heads
  Also returns attention map A (averaged over heads) for the JA incentive.
"""
import functools
from typing import Sequence

import numpy as np
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp

from agents.ja_utils import make_spatial_basis


class JAScannedRNN(nn.Module):
    """Scanned module that performs spatial attention + GRU for each timestep.

    At each step:
      1. Unflatten obs -> grid (H, W, C)
      2. Conv -> features, append spatial basis
      3. Conv -> Keys K, Values V
      4. Queries Q = FFN(h_{t-1})  (top-down from GRU state)
      5. Multi-head attention: A = softmax(Q . K), O = sum(A * V)
      6. GRU(O, h_{t-1}) -> h_t
      7. Return h_t and attention map A (averaged over heads)
    """
    obs_height: int
    obs_width: int
    obs_channels: int = 26
    conv_filters: int = 32
    num_heads: int = 4
    head_features: int = 16
    gru_hidden_dim: int = 64
    spatial_basis_channels: int = 2

    def setup(self):
        self.spatial_basis = make_spatial_basis(self.obs_height, self.obs_width)

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
            carry: GRU hidden state, shape (batch, gru_hidden_dim)
            x: tuple of (obs, dones) where obs is (batch, obs_flat_dim)
               and dones is (batch,)

        Returns:
            new_carry: updated GRU hidden state
            (gru_output, attn_map): gru output for actor/critic, and
                attention map averaged over heads (batch, H, W)
        """
        rnn_state = carry
        obs_flat, dones = x

        batch_size = obs_flat.shape[0]
        h, w, c = self.obs_height, self.obs_width, self.obs_channels
        m, cm = self.num_heads, self.head_features

        # Reset hidden state on episode boundaries
        rnn_state = jnp.where(
            dones[:, np.newaxis],
            self.initialize_carry(batch_size, self.gru_hidden_dim),
            rnn_state,
        )

        # --- 1. Unflatten observation to grid ---
        obs_grid = obs_flat.reshape(batch_size, h, w, c)

        # --- 2. Conv -> features, append spatial basis ---
        features = nn.Conv(
            features=self.conv_filters,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="feature_conv",
        )(obs_grid)  # (batch, H, W, conv_filters)
        features = nn.relu(features)

        # Broadcast spatial basis to batch
        spatial = jnp.broadcast_to(
            self.spatial_basis[None, ...],
            (batch_size, h, w, self.spatial_basis_channels),
        )
        features_with_pos = jnp.concatenate([features, spatial], axis=-1)
        # shape: (batch, H, W, conv_filters + spatial_basis_channels)

        # --- 3. Conv -> Keys and Values ---
        cf_total = self.conv_filters + self.spatial_basis_channels

        keys = nn.Conv(
            features=m * cm,
            kernel_size=(1, 1),
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="key_conv",
        )(features_with_pos)  # (batch, H, W, m*cm)
        keys = keys.reshape(batch_size, h * w, m, cm)  # (batch, HW, m, cm)

        values = nn.Conv(
            features=m * cm,
            kernel_size=(1, 1),
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="value_conv",
        )(features_with_pos)  # (batch, H, W, m*cm)
        values = values.reshape(batch_size, h * w, m, cm)  # (batch, HW, m, cm)

        # --- 4. Queries from GRU hidden state (top-down, goal-directed) ---
        queries = nn.Dense(
            m * cm,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="query_ffn",
        )(rnn_state)  # (batch, m*cm)
        queries = queries.reshape(batch_size, m, cm)  # (batch, m, cm)

        # --- 5. Multi-head spatial attention ---
        # Attention logits: (batch, HW, m) = einsum(batch,HW,m,cm ; batch,m,cm)
        attn_logits = jnp.einsum("bnmc,bmc->bnm", keys, queries)
        # Softmax over spatial locations (n = H*W) per head
        attn_weights = jax.nn.softmax(attn_logits, axis=1)  # (batch, HW, m)

        # Attended output: weighted sum of values
        # (batch, m, cm) = einsum(batch,HW,m ; batch,HW,m,cm)
        attended = jnp.einsum("bnm,bnmc->bmc", attn_weights, values)
        attended_flat = attended.reshape(batch_size, m * cm)  # (batch, m*cm)

        # Attention map averaged over heads for JA incentive
        attn_map = attn_weights.mean(axis=-1)  # (batch, HW)
        attn_map = attn_map.reshape(batch_size, h, w)  # (batch, H, W)

        # --- 6. GRU update ---
        new_rnn_state, gru_out = nn.GRUCell(
            features=self.gru_hidden_dim,
        )(rnn_state, attended_flat)

        return new_rnn_state, (gru_out, attn_map)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class JAActorCritic(nn.Module):
    """Joint Attention Actor-Critic.

    Wraps JAScannedRNN with actor and critic heads.
    Returns attention map alongside policy and value for the JA incentive.
    """
    action_dim: int
    obs_height: int
    obs_width: int
    obs_channels: int = 26
    conv_filters: int = 32
    num_heads: int = 4
    head_features: int = 16
    fc_hidden_dim: int = 64
    gru_hidden_dim: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(self, hidden, x):
        if self.activation == "tanh":
            activation = nn.tanh
        else:
            activation = nn.relu

        obs, dones, avail_actions = x

        # Attention + GRU (scanned over time)
        hidden, (embedding, attn_map) = JAScannedRNN(
            obs_height=self.obs_height,
            obs_width=self.obs_width,
            obs_channels=self.obs_channels,
            conv_filters=self.conv_filters,
            num_heads=self.num_heads,
            head_features=self.head_features,
            gru_hidden_dim=self.gru_hidden_dim,
        )(hidden, (obs, dones))

        # Actor head
        actor_mean = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)

        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        # Critic head
        critic = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1), attn_map
