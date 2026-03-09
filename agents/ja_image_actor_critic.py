"""Joint Attention Actor-Critic network for image observations.

Architecture:
  Shared ResNet encoder extracts spatial features from the observation image.
  Per-agent attention heads (K/V/Q projections, multi-head attention, FC, LSTM)
  process these features independently, producing separate attention maps
  and embeddings for each agent. Per-agent actor and critic projection heads
  produce actions and values.

  The batch dimension is split at the midpoint: first half = agent 0,
  second half = agent 1. This matches the batchify layout used in training
  (batchify stacks agent_0 envs first, then agent_1 envs).
"""
import functools

import numpy as np
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp

from agents.ja_utils import make_sinusoidal_spatial_basis
from agents.resnet_encoder import ResNetEncoder


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


class AgentAttentionHead(nn.Module):
    """Per-agent attention head: K/V/Q projections, multi-head attention, FC, LSTM.

    Takes shared ResNet features (with spatial basis) and the agent's LSTM state,
    computes attention over spatial locations, and updates the LSTM.
    """
    num_heads: int
    head_features: int
    fc_hidden_dim: int
    lstm_hidden_dim: int
    feat_h: int
    feat_w: int

    @nn.compact
    def __call__(self, features_with_pos, lstm_h, lstm_c):
        m, cm = self.num_heads, self.head_features
        fh, fw = self.feat_h, self.feat_w
        batch = features_with_pos.shape[0]

        keys = nn.Conv(
            features=m * cm, kernel_size=(1, 1),
            kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="key_conv",
        )(features_with_pos)
        keys = keys.reshape(batch, fh * fw, m, cm)

        values = nn.Conv(
            features=m * cm, kernel_size=(1, 1),
            kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="value_conv",
        )(features_with_pos)
        values = values.reshape(batch, fh * fw, m, cm)

        own_state = jnp.concatenate([lstm_h, lstm_c], axis=-1)
        queries = nn.Dense(
            m * cm, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            name="query_ffn",
        )(own_state)
        queries = queries.reshape(batch, m, cm)

        attn_logits = jnp.einsum("bnmc,bmc->bnm", keys, queries)
        attn_weights = jax.nn.softmax(attn_logits, axis=1)
        attended = jnp.einsum("bnm,bnmc->bmc", attn_weights, values)
        attended_flat = attended.reshape(batch, m * cm)
        attn_map = attn_weights.mean(axis=-1).reshape(batch, fh, fw)

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

        (new_h, new_c), lstm_out = nn.OptimizedLSTMCell(
            features=self.lstm_hidden_dim,
        )((lstm_h, lstm_c), lstm_input)

        return (new_h, new_c), lstm_out, attn_map


class ProjectionHead(nn.Module):
    """FC -> FC -> linear projection, used for per-agent actor/critic heads."""
    fc_hidden_dim: int
    output_dim: int
    output_init_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.fc_hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(self.output_dim, kernel_init=orthogonal(self.output_init_scale),
                        bias_init=constant(0.0))(x)


class JAImageScannedLSTM(nn.Module):
    """Scanned module: shared ResNet encoder + per-agent attention heads.

    The batch is split at the midpoint: first half goes through agent 0's
    attention head, second half through agent 1's. Each head has independent
    K/V/Q projections, FC layers, and LSTM parameters.
    """
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

    def setup(self):
        self.feat_h, self.feat_w = _compute_resnet_output_dims(
            self.img_height, self.img_width,
            self.conv_stride, self.conv_kernel_size,
            self.conv_padding, self.conv_num_blocks,
        )
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
        n = batch_size // 2
        fh, fw = self.feat_h, self.feat_w

        # Reset LSTM state on episode boundaries
        zero_h, zero_c = self.initialize_carry(batch_size, self.lstm_hidden_dim)
        lstm_h = jnp.where(dones[:, np.newaxis], zero_h, lstm_h)
        lstm_c = jnp.where(dones[:, np.newaxis], zero_c, lstm_c)

        # Unpack flat obs -> image
        img_flat = obs_flat[:, :self._img_flat_dim]
        image = img_flat.reshape(batch_size, self.img_height, self.img_width, 3)

        # Shared ResNet encoder -> (batch, feat_h, feat_w, conv_filters)
        features = ResNetEncoder(
            num_blocks=self.conv_num_blocks,
            filters=self.conv_filters,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
            name="resnet_encoder",
        )(image)

        # Shared spatial basis
        spatial = jnp.broadcast_to(
            self.spatial_basis[None, ...],
            (batch_size, fh, fw, self.spatial_basis_depth),
        )
        features_with_pos = jnp.concatenate([features, spatial], axis=-1)

        # Per-agent attention heads (batch split: [:n] = agent 0, [n:] = agent 1)
        head_kwargs = dict(
            num_heads=self.num_heads,
            head_features=self.head_features,
            fc_hidden_dim=self.fc_hidden_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            feat_h=fh,
            feat_w=fw,
        )

        (new_h_0, new_c_0), out_0, attn_0 = AgentAttentionHead(
            **head_kwargs, name="agent0_head",
        )(features_with_pos[:n], lstm_h[:n], lstm_c[:n])

        (new_h_1, new_c_1), out_1, attn_1 = AgentAttentionHead(
            **head_kwargs, name="agent1_head",
        )(features_with_pos[n:], lstm_h[n:], lstm_c[n:])

        # Recombine along batch dimension
        new_h = jnp.concatenate([new_h_0, new_h_1], axis=0)
        new_c = jnp.concatenate([new_c_0, new_c_1], axis=0)
        lstm_out = jnp.concatenate([out_0, out_1], axis=0)
        attn_map = jnp.concatenate([attn_0, attn_1], axis=0)

        return (new_h, new_c), (lstm_out, attn_map)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return (
            jnp.zeros((batch_size, hidden_size)),
            jnp.zeros((batch_size, hidden_size)),
        )


class JAImageActorCritic(nn.Module):
    """Joint Attention Actor-Critic with shared ResNet, per-agent heads.

    Expects batch entries for agent 0 in the first half and agent 1 in the
    second half. Each agent has independent attention, LSTM, and projection
    parameters while sharing the ResNet backbone.
    """
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

    @nn.compact
    def __call__(self, hidden, x):
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
        )

        # Actor path
        actor_lstm_state, (actor_embed, attn_map) = JAImageScannedLSTM(
            **rnn_kwargs, name="actor_lstm",
        )(actor_lstm_state, (obs, dones))

        # Per-agent actor projections
        n = actor_embed.shape[-2] // 2
        embed_0, embed_1 = actor_embed[..., :n, :], actor_embed[..., n:, :]

        logits_0 = ProjectionHead(
            self.fc_hidden_dim, self.action_dim, 0.01, name="agent0_actor",
        )(embed_0)
        logits_1 = ProjectionHead(
            self.fc_hidden_dim, self.action_dim, 0.01, name="agent1_actor",
        )(embed_1)
        action_logits = jnp.concatenate([logits_0, logits_1], axis=-2)

        unavail_actions = 1 - avail_actions
        action_logits = action_logits - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        # Critic path
        critic_lstm_state, (critic_embed, _) = JAImageScannedLSTM(
            **rnn_kwargs, name="critic_lstm",
        )(critic_lstm_state, (obs, dones))

        cembed_0, cembed_1 = critic_embed[..., :n, :], critic_embed[..., n:, :]
        val_0 = ProjectionHead(
            self.fc_hidden_dim, 1, 1.0, name="agent0_critic",
        )(cembed_0)
        val_1 = ProjectionHead(
            self.fc_hidden_dim, 1, 1.0, name="agent1_critic",
        )(cembed_1)
        value = jnp.concatenate([val_0, val_1], axis=-2)

        new_hidden = (actor_lstm_state, critic_lstm_state)
        return new_hidden, pi, jnp.squeeze(value, axis=-1), attn_map
