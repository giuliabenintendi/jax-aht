"""Joint Attention Actor-Critic network for image observations.

Same architecture as ja_actor_critic.py but replaces the symbolic 26-channel
observation path with a ResNet encoder, following the google-research
`use_stacks=True` path.

Per-agent architecture:
  obs (flat) -> unpack image (H*7, W*7, 3) + scalars (6,)
  Image -> Stack(conv_filters//2) -> Stack(conv_filters) -> ReLU -> features F
  F + sinusoidal spatial basis -> 1x1 Conv -> Keys K, Values V
  Q = Dense(concat(h_ego, h_partner))
  Multi-head attention: softmax(Q . K) -> attended O
  Scalars: direction one_hot(4) x2 -> Dense(5), position (4,) -> Dense(5)
  LSTM(concat(O, dir_embed, pos_embed), (h, c)) -> h_t
  Dense(64) -> act -> Dense(64) -> act -> output
"""
import functools
import math

import numpy as np
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp

from agents.ja_utils import make_sinusoidal_spatial_basis


class _ResBlock(nn.Module):
    """Single residual block: ReLU -> Conv(3x3) -> ReLU -> Conv(3x3) + skip."""
    filters: int

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.relu(x)
        y = nn.Conv(
            features=self.filters, kernel_size=(3, 3), padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
        )(y)
        y = nn.relu(y)
        y = nn.Conv(
            features=self.filters, kernel_size=(3, 3), padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
        )(y)
        return y + residual


class _Stack(nn.Module):
    """ResNet stack: Conv(3x3) -> MaxPool(3x3, stride=2) -> N residual blocks."""
    filters: int
    num_blocks: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.filters, kernel_size=(3, 3), padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        for i in range(self.num_blocks):
            x = _ResBlock(filters=self.filters, name=f"res_{i}")(x)
        return x


class JAImageScannedLSTM(nn.Module):
    """Scanned module: ResNet encoder + spatial attention + LSTM.

    At each timestep:
      1. Unpack flat obs -> image (H*7, W*7, 3) + scalars (6,)
      2. Image -> 2 ResNet stacks -> ReLU -> features F (H', W', conv_filters)
      3. F + spatial basis -> 1x1 Conv -> Keys K, Values V
      4. Q = Dense(concat(h_ego, h_partner))
      5. Multi-head spatial attention -> attended O
      6. Scalars: direction one_hot(4) -> Dense(5), position -> Dense(5)
      7. LSTM(concat(O, dir_embed, pos_embed)) -> (h_t, c_t)
      8. Return (h_t, c_t) and attention map (averaged over heads)
    """
    img_height: int   # H*tile_size (pixels)
    img_width: int    # W*tile_size (pixels)
    num_scalars: int = 6
    conv_filters: int = 64
    num_heads: int = 4
    head_features: int = 16
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
    spatial_basis_depth: int = 8
    scalar_embed_dim: int = 5

    def setup(self):
        # Spatial dims after two MaxPool(stride=2): ceil(dim/4)
        feat_h = math.ceil(self.img_height / 4)
        feat_w = math.ceil(self.img_width / 4)
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.spatial_basis = make_sinusoidal_spatial_basis(
            feat_h, feat_w, self.spatial_basis_depth,
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
        obs_flat, dones, partner_hstate = x

        batch_size = obs_flat.shape[0]
        m, cm = self.num_heads, self.head_features
        fh, fw = self.feat_h, self.feat_w

        # Reset LSTM state on episode boundaries
        zero_h, zero_c = self.initialize_carry(batch_size, self.lstm_hidden_dim)
        lstm_h = jnp.where(dones[:, np.newaxis], zero_h, lstm_h)
        lstm_c = jnp.where(dones[:, np.newaxis], zero_c, lstm_c)

        # --- 1. Unpack flat obs -> image + scalars ---
        img_flat = obs_flat[:, :self._img_flat_dim]
        scalars = obs_flat[:, self._img_flat_dim:]  # (batch, 6)
        image = img_flat.reshape(batch_size, self.img_height, self.img_width, 3)

        # --- 2. ResNet encoder: two stacks ---
        features = _Stack(
            filters=self.conv_filters // 2, num_blocks=2, name="stack_0",
        )(image)
        features = _Stack(
            filters=self.conv_filters, num_blocks=2, name="stack_1",
        )(features)
        features = nn.relu(features)  # (batch, fh, fw, conv_filters)

        # --- 3. Append spatial basis, compute K and V via 1x1 conv ---
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

        # --- 4. Cross-agent query ---
        query_input = jnp.concatenate([lstm_h, partner_hstate], axis=-1)
        queries = nn.Dense(
            m * cm, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="query_ffn",
        )(query_input)
        queries = queries.reshape(batch_size, m, cm)

        # --- 5. Multi-head spatial attention (no sqrt scaling) ---
        attn_logits = jnp.einsum("bnmc,bmc->bnm", keys, queries)
        attn_weights = jax.nn.softmax(attn_logits, axis=1)  # (batch, HW, m)

        attended = jnp.einsum("bnm,bnmc->bmc", attn_weights, values)
        attended_flat = attended.reshape(batch_size, m * cm)

        attn_map = attn_weights.mean(axis=-1).reshape(batch_size, fh, fw)

        # --- 6. Scalar features ---
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

        # --- 7. FC layers before LSTM (reference code: input_fc_layer_params) ---
        lstm_input = jnp.concatenate([attended_flat, dir_embed, pos_embed], axis=-1)
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

        # --- 8. LSTM update ---
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
    (ResNet encoder) instead of JAScannedLSTM (Conv on symbolic channels).
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
        obs, dones, avail_actions, partner_hstate = x

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
        )(actor_lstm_state, (obs, dones, partner_hstate))

        # FFN after LSTM (paper diagram: LSTM → FFN → Action)
        actor_out = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0), name="actor_fc1",
        )(actor_embed)
        actor_out = nn.relu(actor_out)
        actor_out = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0), name="actor_fc2",
        )(actor_out)
        actor_out = nn.relu(actor_out)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
            name="actor_proj",
        )(actor_out)

        unavail_actions = 1 - avail_actions
        action_logits = action_logits - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        # --- Critic path ---
        critic_lstm_state, (critic_embed, _) = JAImageScannedLSTM(
            **rnn_kwargs, name="critic_lstm",
        )(critic_lstm_state, (obs, dones, partner_hstate))

        critic_out = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0), name="critic_fc1",
        )(critic_embed)
        critic_out = nn.relu(critic_out)
        critic_out = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0), name="critic_fc2",
        )(critic_out)
        critic_out = nn.relu(critic_out)
        value = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="critic_proj",
        )(critic_out)

        new_hidden = (actor_lstm_state, critic_lstm_state)
        return new_hidden, pi, jnp.squeeze(value, axis=-1), attn_map
