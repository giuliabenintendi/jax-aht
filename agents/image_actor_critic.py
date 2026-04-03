"""Image Actor-Critic network (no attention baseline).

Same ResNet encoder as `ja_image_actor_critic.py` but without spatial
attention. The image trunk is shared between actor and critic:

  obs (flat) -> unpack image (H_px, W_px, 3)
  Image -> ResNet encoder -> flatten -> FC -> FC -> shared LSTM
  shared embedding -> actor head
  shared embedding -> critic head
"""
import functools

import numpy as np
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp

from agents.action_masking import mask_action_logits
from agents.resnet_encoder import ResNetEncoder


class ImageScannedLSTM(nn.Module):
    """Scanned shared image trunk: ResNet encoder + flatten + LSTM."""
    img_height: int
    img_width: int
    conv_filters: int = 32
    conv_num_blocks: int = 4
    conv_kernel_size: int = 3
    conv_stride: int = 2
    conv_padding: str = "SAME"
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64

    def setup(self):
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

        # Reset LSTM state on episode boundaries
        zero_h, zero_c = self.initialize_carry(batch_size, self.lstm_hidden_dim)
        lstm_h = jnp.where(dones[:, np.newaxis], zero_h, lstm_h)
        lstm_c = jnp.where(dones[:, np.newaxis], zero_c, lstm_c)

        # Unpack flat obs -> image
        img_flat = obs_flat[:, :self._img_flat_dim]
        image = img_flat.reshape(batch_size, self.img_height, self.img_width, 3)

        # ResNet encoder -> (batch, H_out, W_out, conv_filters)
        features = ResNetEncoder(
            num_blocks=self.conv_num_blocks,
            filters=self.conv_filters,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
            name="resnet_encoder",
        )(image)

        # Flatten spatial features
        features_flat = features.reshape(batch_size, -1)

        # FC layers before LSTM
        lstm_input = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="input_fc1",
        )(features_flat)
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

    Uses a shared image encoder and recurrent state, with separate policy and
    value heads on top of the shared embedding.
    """
    action_dim: int
    img_height: int
    img_width: int
    conv_filters: int = 32
    conv_num_blocks: int = 4
    conv_kernel_size: int = 3
    conv_stride: int = 2
    conv_padding: str = "SAME"
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x

        rnn_kwargs = dict(
            img_height=self.img_height,
            img_width=self.img_width,
            conv_filters=self.conv_filters,
            conv_num_blocks=self.conv_num_blocks,
            conv_kernel_size=self.conv_kernel_size,
            conv_stride=self.conv_stride,
            conv_padding=self.conv_padding,
            fc_hidden_dim=self.fc_hidden_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
        )

        hidden, (shared_embed,) = ImageScannedLSTM(
            **rnn_kwargs, name="shared_lstm",
        )(hidden, (obs, dones))

        actor_out = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0), name="actor_fc1",
        )(shared_embed)
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

        action_logits = mask_action_logits(action_logits, avail_actions)
        pi = distrax.Categorical(logits=action_logits)

        critic_out = nn.Dense(
            self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0), name="critic_fc1",
        )(shared_embed)
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

        return hidden, pi, jnp.squeeze(value, axis=-1)
