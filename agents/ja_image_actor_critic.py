"""Joint Attention Actor-Critic network for image observations.

Same architecture as ja_actor_critic.py but operates on pixel observations
(H_px, W_px, 3) instead of symbolic (H, W, 26) grids.

Matches the google-research `use_stacks=False` default: a single Conv layer
processes the image (no ResNet, no downsampling), then spatial attention + LSTM.

Per-agent architecture:
  obs (flat) -> unpack image (H_px, W_px, 3) + scalars (num_scalars,)
  Image -> Conv(3x3, conv_filters, SAME) -> ReLU -> features F
  F + sinusoidal spatial basis -> 1x1 Conv -> Keys K, Values V
  Q = Dense(concat(h, c)) — query from own LSTM state
  Multi-head attention: softmax(Q . K) -> attended O
  Scalars: direction one_hot(4) x2 -> Dense(5), position (4,) -> Dense(5)
  Concat(O, dir_embed, pos_embed) -> FC -> FC -> LSTM -> projection
"""
import functools

import numpy as np
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp

from agents.ja_utils import make_sinusoidal_spatial_basis


class JAImageScannedLSTM(nn.Module):
    """Scanned module: Conv encoder + spatial attention + LSTM.

    At each timestep:
      1. Unpack flat obs -> image (H_px, W_px, 3) + scalars
      2. Image -> Conv(3x3, conv_filters) -> ReLU -> features F
      3. F + spatial basis -> 1x1 Conv -> Keys K, Values V
      4. Q = Dense(concat(h, c)) — query from own LSTM state
      5. Multi-head spatial attention -> attended O
      6. Scalars: direction one_hot(4) -> Dense(5), position -> Dense(5)
      7. Concat(O, dir_embed, pos_embed) -> FC -> FC -> LSTM -> (h_t, c_t)
      8. Return (h_t, c_t) and attention map (averaged over heads)
    """
    img_height: int   # H_px (pixels)
    img_width: int    # W_px (pixels)
    num_scalars: int = 6
    conv_filters: int = 64
    num_heads: int = 4
    head_features: int = 16
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
    spatial_basis_depth: int = 8
    scalar_embed_dim: int = 5

    def setup(self):
        # No downsampling — feature map keeps original spatial dims
        self.feat_h = self.img_height
        self.feat_w = self.img_width
        self.spatial_basis = make_sinusoidal_spatial_basis(
            self.feat_h, self.feat_w, self.spatial_basis_depth,
        )
        self._img_flat_dim = self.img_height * self.img_width * 3

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        lstm_h, lstm_c = carry
        obs_flat, dones = x

        batch_size = obs_flat.shape[0]
        m, cm = self.num_heads, self.head_features
        fh, fw = self.feat_h, self.feat_w

        # Reset LSTM state on episode boundaries
        zero_h, zero_c = self.initialize_carry(batch_size, self.lstm_hidden_dim)
        lstm_h = jnp.where(dones[:, np.newaxis], zero_h, lstm_h)
        lstm_c = jnp.where(dones[:, np.newaxis], zero_c, lstm_c)

        # --- Unpack flat obs -> image (+ optional scalars) ---
        img_flat = obs_flat[:, :self._img_flat_dim]
        image = img_flat.reshape(batch_size, self.img_height, self.img_width, 3)

        # --- Single Conv encoder (matches google-research use_stacks=False) ---
        features = nn.Conv(
            features=self.conv_filters,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="feature_conv",
        )(image)
        features = nn.relu(features)  # (batch, fh, fw, conv_filters)

        # --- Append spatial basis, compute K and V via 1x1 conv ---
        spatial = jnp.broadcast_to(
            self.spatial_basis[None, ...],
            (batch_size, fh, fw, self.spatial_basis_depth),
        )
        features_with_pos = jnp.concatenate([features, spatial], axis=-1)

        keys = nn.Conv(
            features=m * cm, kernel_size=(1, 1),
            kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="key_conv",
        )(features_with_pos)
        keys = keys.reshape(batch_size, fh * fw, m, cm)

        values = nn.Conv(
            features=m * cm, kernel_size=(1, 1),
            kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="value_conv",
        )(features_with_pos)
        values = values.reshape(batch_size, fh * fw, m, cm)

        # --- Query from own LSTM state: Q = Dense(concat(h, c)) ---
        own_state = jnp.concatenate([lstm_h, lstm_c], axis=-1)
        queries = nn.Dense(
            m * cm, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="query_ffn",
        )(own_state)
        queries = queries.reshape(batch_size, m, cm)

        # --- Multi-head spatial attention (no sqrt scaling) ---
        attn_logits = jnp.einsum("bnmc,bmc->bnm", keys, queries)
        attn_weights = jax.nn.softmax(attn_logits, axis=1)  # (batch, HW, m)

        attended = jnp.einsum("bnm,bnmc->bmc", attn_weights, values)
        attended_flat = attended.reshape(batch_size, m * cm)

        attn_map = attn_weights.mean(axis=-1).reshape(batch_size, fh, fw)

        # --- Scalar features (skipped when num_scalars == 0, e.g. FOV obs) ---
        lstm_parts = [attended_flat]
        if self.num_scalars > 0:
            scalars = obs_flat[:, self._img_flat_dim:]  # (batch, num_scalars)
            # scalars: [ego_dir, ego_x, ego_y, partner_dir, partner_x, partner_y]
            ego_dir_idx = scalars[:, 0].astype(jnp.int32)
            partner_dir_idx = scalars[:, 3].astype(jnp.int32)
            ego_dir_onehot = jax.nn.one_hot(ego_dir_idx, 4)
            partner_dir_onehot = jax.nn.one_hot(partner_dir_idx, 4)
            direction = jnp.concatenate([ego_dir_onehot, partner_dir_onehot], axis=-1)  # (batch, 8)
            dir_embed = nn.Dense(
                self.scalar_embed_dim,
                kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                name="dir_embed",
            )(direction)

            pos_features = scalars[:, jnp.array([1, 2, 4, 5])]  # (batch, 4)
            pos_embed = nn.Dense(
                self.scalar_embed_dim,
                kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                name="pos_embed",
            )(pos_features)
            lstm_parts.extend([dir_embed, pos_embed])

        # --- FC layers before LSTM ---
        lstm_input = jnp.concatenate(lstm_parts, axis=-1)
        lstm_input = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="input_fc1",
        )(lstm_input)
        lstm_input = nn.relu(lstm_input)
        lstm_input = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="input_fc2",
        )(lstm_input)
        lstm_input = nn.relu(lstm_input)

        # --- LSTM update ---
        new_carry, lstm_out = nn.OptimizedLSTMCell(
            features=self.lstm_hidden_dim,
        )((lstm_h, lstm_c), lstm_input)
        new_h, new_c = new_carry

        return (new_h, new_c), (lstm_out, attn_map)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return (
            jnp.zeros((batch_size, hidden_size)),
            jnp.zeros((batch_size, hidden_size)),
        )


class JAImageActorCritic(nn.Module):
    """Joint Attention Actor-Critic for image observations.

    Identical structure to JAActorCritic but uses JAImageScannedLSTM
    (single Conv on pixel input) instead of JAScannedLSTM (Conv on symbolic channels).
    """
    action_dim: int
    img_height: int
    img_width: int
    num_scalars: int = 6
    conv_filters: int = 64
    num_heads: int = 4
    head_features: int = 16
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
    spatial_basis_depth: int = 8
    scalar_embed_dim: int = 5

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x

        actor_lstm_state, critic_lstm_state = hidden

        rnn_kwargs = dict(
            img_height=self.img_height,
            img_width=self.img_width,
            num_scalars=self.num_scalars,
            conv_filters=self.conv_filters,
            num_heads=self.num_heads,
            head_features=self.head_features,
            fc_hidden_dim=self.fc_hidden_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            spatial_basis_depth=self.spatial_basis_depth,
            scalar_embed_dim=self.scalar_embed_dim,
        )

        # --- Actor path ---
        actor_lstm_state, (actor_embed, attn_map) = JAImageScannedLSTM(
            **rnn_kwargs, name="actor_lstm",
        )(actor_lstm_state, (obs, dones))

        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
            name="actor_proj",
        )(actor_embed)

        unavail_actions = 1 - avail_actions
        action_logits = action_logits - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        # --- Critic path ---
        critic_lstm_state, (critic_embed, _) = JAImageScannedLSTM(
            **rnn_kwargs, name="critic_lstm",
        )(critic_lstm_state, (obs, dones))

        value = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="critic_proj",
        )(critic_embed)

        new_hidden = (actor_lstm_state, critic_lstm_state)
        return new_hidden, pi, jnp.squeeze(value, axis=-1), attn_map
