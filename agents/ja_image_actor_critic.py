"""Joint Attention Actor-Critic network for image observations.

Per-agent architecture:
  obs (flat) -> unpack image (H_px, W_px, num_channels)
  Image -> ResNet encoder -> features F  (H_out, W_out, filters)
  F + sinusoidal spatial basis -> 1x1 Conv -> Keys K, Values V
  Q = Dense(concat(h, c)) — query from own LSTM state
  Multi-head attention: softmax(Q . K) -> attended O
  Concat(O) -> FC -> FC -> LSTM -> projection

When FEED_OTHER_ATTN is enabled, num_channels=4: the 4th channel is the
other agent's previous attention map (nearest-neighbor upsampled to pixel
resolution), providing a human-like joint attention signal.
"""
import functools

import numpy as np
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp

from agents.action_masking import mask_action_logits
from agents.ja_utils import make_sinusoidal_spatial_basis
from agents.resnet_encoder import ResNetEncoder


class CrossAgentAttention(nn.Module):
    """Cross-attention between an agent's embedding and its partner's.

    With only 2 agents (self + partner), the softmax over a single key
    collapses to a sigmoid gate on the partner's value projection.
    """
    embed_dim: int

    @nn.compact
    def __call__(self, own_embed, partner_embed):
        q = nn.Dense(self.embed_dim, kernel_init=orthogonal(1.0),
                     bias_init=constant(0.0), name="xattn_q")(own_embed)
        k = nn.Dense(self.embed_dim, kernel_init=orthogonal(1.0),
                     bias_init=constant(0.0), name="xattn_k")(partner_embed)
        v = nn.Dense(self.embed_dim, kernel_init=orthogonal(1.0),
                     bias_init=constant(0.0), name="xattn_v")(partner_embed)

        score = jnp.sum(q * k, axis=-1, keepdims=True) / np.sqrt(self.embed_dim)
        gate = jax.nn.sigmoid(score)
        cross_out = gate * v

        out = nn.Dense(self.embed_dim, kernel_init=orthogonal(1.0),
                       bias_init=constant(0.0), name="xattn_out")(cross_out)
        return own_embed + out


def _compute_resnet_output_dims(h, w, stride, kernel_size, padding, num_blocks):
    """Compute spatial dims after ResNet encoder (initial conv + first block downsample)."""
    for _ in range(1 + min(1, num_blocks)):
        if padding == "SAME":
            h = int(np.ceil(h / stride))
            w = int(np.ceil(w / stride))
        else:
            h = (h - kernel_size) // stride + 1
            w = (w - kernel_size) // stride + 1
    return h, w


class JAImageScannedLSTM(nn.Module):
    """Scanned module: ResNet encoder + spatial attention + LSTM."""
    img_height: int
    img_width: int
    conv_filters: int = 32
    conv_num_blocks: int = 4
    conv_kernel_size: int = 3
    conv_stride: int = 2
    conv_padding: str = "SAME"
    num_heads: int = 4
    head_features: int = 16
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
    spatial_basis_depth: int = 8
    num_channels: int = 3
    message_dim: int = 0  # >0 enables communication (partner message one-hot appended to obs)
    scalar_dim: int = 0   # >0 appends extra scalar features after any message suffix
    scalar_embed_dim: int = 5
    cross_agent_attn: bool = False

    def setup(self):
        self.feat_h, self.feat_w = _compute_resnet_output_dims(
            self.img_height, self.img_width,
            self.conv_stride, self.conv_kernel_size,
            self.conv_padding, self.conv_num_blocks,
        )
        self.spatial_basis = make_sinusoidal_spatial_basis(
            self.feat_h, self.feat_w, self.spatial_basis_depth,
        )
        self._img_flat_dim = self.img_height * self.img_width * self.num_channels

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
        if self.cross_agent_attn:
            obs_flat, dones, partner_embed = x
        else:
            obs_flat, dones = x

        batch_size = obs_flat.shape[0]
        m, cm = self.num_heads, self.head_features
        fh, fw = self.feat_h, self.feat_w

        # Reset LSTM state on episode boundaries
        zero_h, zero_c = self.initialize_carry(batch_size, self.lstm_hidden_dim)
        lstm_h = jnp.where(dones[:, np.newaxis], zero_h, lstm_h)
        lstm_c = jnp.where(dones[:, np.newaxis], zero_c, lstm_c)

        # Unpack flat obs -> image
        img_flat = obs_flat[:, :self._img_flat_dim]
        image = img_flat.reshape(batch_size, self.img_height, self.img_width, self.num_channels)

        # ResNet encoder -> (batch, feat_h, feat_w, conv_filters)
        features = ResNetEncoder(
            num_blocks=self.conv_num_blocks,
            filters=self.conv_filters,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
            name="resnet_encoder",
        )(image)

        # Append spatial basis, compute K and V via 1x1 conv
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

        # Query from own LSTM state
        own_state = jnp.concatenate([lstm_h, lstm_c], axis=-1)
        queries = nn.Dense(
            m * cm, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="query_ffn",
        )(own_state)
        queries = queries.reshape(batch_size, m, cm)

        # Multi-head spatial attention
        attn_logits = jnp.einsum("bnmc,bmc->bnm", keys, queries)
        attn_weights = jax.nn.softmax(attn_logits, axis=1)

        attended = jnp.einsum("bnm,bnmc->bmc", attn_weights, values)
        attended_flat = attended.reshape(batch_size, m * cm)

        attn_map = attn_weights.mean(axis=-1).reshape(batch_size, fh, fw)

        suffix_parts = []
        suffix_start = self._img_flat_dim

        # Concatenate partner's message one-hot if communication is enabled
        if self.message_dim > 0:
            msg_input = obs_flat[:, suffix_start:suffix_start + self.message_dim]
            suffix_parts.append(msg_input)
            suffix_start += self.message_dim

        if self.scalar_dim > 0:
            scalar_input = obs_flat[:, suffix_start:suffix_start + self.scalar_dim]
            scalar_embed = nn.Dense(
                self.scalar_embed_dim,
                kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
                name="scalar_embed",
            )(scalar_input)
            suffix_parts.append(scalar_embed)

        if suffix_parts:
            attended_flat = jnp.concatenate([attended_flat] + suffix_parts, axis=-1)

        # Save own embedding before cross-attention (for partner to use next step)
        own_embed = attended_flat

        if self.cross_agent_attn:
            attended_flat = CrossAgentAttention(
                embed_dim=m * cm,
                name="cross_agent_attn",
            )(attended_flat, jax.lax.stop_gradient(partner_embed))

        # FC layers before LSTM
        lstm_input = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            name="input_fc1",
        )(attended_flat)
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

        if self.cross_agent_attn:
            return (new_h, new_c), (lstm_out, attn_map, jax.lax.stop_gradient(own_embed))
        return (new_h, new_c), (lstm_out, attn_map)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return (
            jnp.zeros((batch_size, hidden_size)),
            jnp.zeros((batch_size, hidden_size)),
        )


class JAImageActorCritic(nn.Module):
    """Joint Attention Actor-Critic for image observations with ResNet encoder."""
    action_dim: int
    img_height: int
    img_width: int
    conv_filters: int = 32
    conv_num_blocks: int = 4
    conv_kernel_size: int = 3
    conv_stride: int = 2
    conv_padding: str = "SAME"
    num_heads: int = 4
    head_features: int = 16
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
    spatial_basis_depth: int = 8
    num_channels: int = 3
    message_dim: int = 0  # >0 enables communication (partner message input via obs)
    scalar_dim: int = 0
    scalar_embed_dim: int = 5
    cross_agent_attn: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        if self.cross_agent_attn:
            obs, dones, avail_actions, partner_embed_actor, partner_embed_critic = x
        else:
            obs, dones, avail_actions = x

        actor_lstm_state, critic_lstm_state = hidden

        rnn_kwargs = dict(
            img_height=self.img_height,
            img_width=self.img_width,
            conv_filters=self.conv_filters,
            conv_num_blocks=self.conv_num_blocks,
            conv_kernel_size=self.conv_kernel_size,
            conv_stride=self.conv_stride,
            conv_padding=self.conv_padding,
            num_heads=self.num_heads,
            head_features=self.head_features,
            fc_hidden_dim=self.fc_hidden_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            spatial_basis_depth=self.spatial_basis_depth,
            num_channels=self.num_channels,
            message_dim=self.message_dim,
            scalar_dim=self.scalar_dim,
            scalar_embed_dim=self.scalar_embed_dim,
            cross_agent_attn=self.cross_agent_attn,
        )

        # Actor path
        if self.cross_agent_attn:
            actor_lstm_state, (actor_embed, attn_map, actor_own_embed) = JAImageScannedLSTM(
                **rnn_kwargs, name="actor_lstm",
            )(actor_lstm_state, (obs, dones, partner_embed_actor))
        else:
            actor_lstm_state, (actor_embed, attn_map) = JAImageScannedLSTM(
                **rnn_kwargs, name="actor_lstm",
            )(actor_lstm_state, (obs, dones))

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

        action_logits = mask_action_logits(action_logits, avail_actions)
        pi = distrax.Categorical(logits=action_logits)

        # Critic path
        if self.cross_agent_attn:
            critic_lstm_state, (critic_embed, _, critic_own_embed) = JAImageScannedLSTM(
                **rnn_kwargs, name="critic_lstm",
            )(critic_lstm_state, (obs, dones, partner_embed_critic))
        else:
            critic_lstm_state, (critic_embed, _) = JAImageScannedLSTM(
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
        if self.cross_agent_attn:
            return new_hidden, pi, jnp.squeeze(value, axis=-1), attn_map, actor_own_embed, critic_own_embed
        return new_hidden, pi, jnp.squeeze(value, axis=-1), attn_map
