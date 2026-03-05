"""Image Actor-Critic network (no attention baseline).

Same single-Conv encoder as ja_image_actor_critic.py but without spatial
attention — features are flattened directly into the LSTM.

Per-agent architecture:
  obs (flat) -> unpack image (H_px, W_px, 3) + scalars (num_scalars,)
  Image -> Conv(3x3, conv_filters, SAME) -> ReLU -> flatten
  Scalars: direction one_hot(4) x2 -> Dense(5), position (4,) -> Dense(5)
  concat(flat_features, dir_embed, pos_embed) -> Dense(64) -> ReLU -> Dense(64) -> ReLU
  LSTM(fc_out, (h, c)) -> h_t -> projection head
"""
import functools

import numpy as np
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp


class ImageScannedLSTM(nn.Module):
    """Scanned module: Conv encoder + flatten + LSTM (no attention)."""
    img_height: int
    img_width: int
    num_scalars: int = 6
    conv_filters: int = 64
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
    scalar_embed_dim: int = 5

    def setup(self):
        # No downsampling — feature map keeps original spatial dims
        self.feat_h = self.img_height
        self.feat_w = self.img_width
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
        fh, fw = self.feat_h, self.feat_w

        # Reset LSTM state on episode boundaries
        zero_h, zero_c = self.initialize_carry(batch_size, self.lstm_hidden_dim)
        lstm_h = jnp.where(dones[:, np.newaxis], zero_h, lstm_h)
        lstm_c = jnp.where(dones[:, np.newaxis], zero_c, lstm_c)

        # Unpack flat obs -> image (+ optional scalars)
        img_flat = obs_flat[:, :self._img_flat_dim]
        image = img_flat.reshape(batch_size, self.img_height, self.img_width, 3)

        # Single Conv encoder (matches google-research use_stacks=False)
        features = nn.Conv(
            features=self.conv_filters,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="feature_conv",
        )(image)
        features = nn.relu(features)  # (batch, fh, fw, conv_filters)

        # Flatten spatial features
        features_flat = features.reshape(batch_size, fh * fw * self.conv_filters)

        # Scalar features (skipped when num_scalars == 0, e.g. FOV obs)
        lstm_parts = [features_flat]
        if self.num_scalars > 0:
            scalars = obs_flat[:, self._img_flat_dim:]  # (batch, num_scalars)
            ego_dir_idx = scalars[:, 0].astype(jnp.int32)
            partner_dir_idx = scalars[:, 3].astype(jnp.int32)
            ego_dir_onehot = jax.nn.one_hot(ego_dir_idx, 4)
            partner_dir_onehot = jax.nn.one_hot(partner_dir_idx, 4)
            direction = jnp.concatenate([ego_dir_onehot, partner_dir_onehot], axis=-1)
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

        # FC layers before LSTM
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

        # LSTM update
        new_carry, lstm_out = nn.OptimizedLSTMCell(
            features=self.lstm_hidden_dim,
        )((lstm_h, lstm_c), lstm_input)
        new_h, new_c = new_carry

        return (new_h, new_c), (lstm_out,)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return (
            jnp.zeros((batch_size, hidden_size)),
            jnp.zeros((batch_size, hidden_size)),
        )


class ImageActorCritic(nn.Module):
    """Image Actor-Critic without attention.

    Dual-path (actor LSTM + critic LSTM, no shared weights).
    FC layers are inside the ScannedLSTM (before LSTM), matching the reference
    code's input_fc_layer_params. LSTM output goes directly to projection head.
    Input tuple: (obs, dones, avail_actions).
    """
    action_dim: int
    img_height: int
    img_width: int
    num_scalars: int = 6
    conv_filters: int = 64
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
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
            fc_hidden_dim=self.fc_hidden_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            scalar_embed_dim=self.scalar_embed_dim,
        )

        # Actor path
        actor_lstm_state, (actor_embed,) = ImageScannedLSTM(
            **rnn_kwargs, name="actor_lstm",
        )(actor_lstm_state, (obs, dones))

        # FFN after LSTM (paper diagram: LSTM -> FFN -> Action)
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

        # Critic path
        critic_lstm_state, (critic_embed,) = ImageScannedLSTM(
            **rnn_kwargs, name="critic_lstm",
        )(critic_lstm_state, (obs, dones))

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
        return new_hidden, pi, jnp.squeeze(value, axis=-1)
