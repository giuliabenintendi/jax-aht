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
    """Flamingo-style gated cross-attention over partner's spatial features.

    Q from own embedding, K/V from partner's spatial feature map (H*W positions).
    Multi-head scaled dot-product attention, zero-init gated residual, followed
    by a gated FFN. Adapted from OpenFlamingo's GatedCrossAttentionBlock.

    The partner_features tensor carries sinusoidal spatial basis, so the
    weighted-sum output encodes positional information explicitly.
    """
    embed_dim: int
    num_heads: int = 4
    head_dim: int = 16
    ff_mult: int = 4

    @nn.compact
    def __call__(self, own_embed, partner_features):
        """
        Args:
            own_embed: (batch, embed_dim) — agent's post-attention embedding
            partner_features: (batch, H*W, feat_dim) — partner's spatial features
        """
        if partner_features.ndim != 3:
            raise ValueError(
                "partner_features must have shape (batch, positions, feat_dim); "
                f"got ndim={partner_features.ndim} and shape={partner_features.shape}"
            )

        m, d = self.num_heads, self.head_dim
        inner_dim = m * d
        scale = d ** -0.5

        # LayerNorm on query input (Flamingo pattern)
        x_norm = nn.LayerNorm(name="xattn_ln")(own_embed)

        # Q from own embedding, K/V from partner's spatial features
        q = nn.Dense(inner_dim, use_bias=False, name="xattn_q")(x_norm)
        kv = nn.Dense(inner_dim * 2, use_bias=False, name="xattn_kv")(partner_features)
        k, v = jnp.split(kv, 2, axis=-1)

        # Reshape for multi-head: q (batch, m, d), k/v (batch, HW, m, d)
        q = q.reshape(-1, m, d)
        k = k.reshape(-1, k.shape[1], m, d)
        v = v.reshape(-1, v.shape[1], m, d)

        # Scaled dot-product attention over spatial positions
        attn_logits = jnp.einsum("bmd,bnmd->bmn", q, k) * scale
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # (batch, m, HW)

        attn_out = jnp.einsum("bmn,bnmd->bmd", attn_weights, v)
        attn_out = attn_out.reshape(-1, inner_dim)
        attn_out = nn.Dense(self.embed_dim, use_bias=False, name="xattn_out")(attn_out)

        # Flamingo-style learned residual gate: starts as near-identity.
        attn_gate = self.param("attn_gate", nn.initializers.zeros, (1,))

        # Gated residual
        x = own_embed + jnp.tanh(attn_gate) * attn_out

        # Gated FFN (Flamingo pattern)
        ff_dim = self.embed_dim * self.ff_mult
        ff_out = nn.LayerNorm(name="ff_ln")(x)
        ff_out = nn.Dense(ff_dim, use_bias=False, name="ff_up")(ff_out)
        ff_out = nn.gelu(ff_out)
        ff_out = nn.Dense(self.embed_dim, use_bias=False, name="ff_down")(ff_out)
        ff_gate = self.param("ff_gate", nn.initializers.zeros, (1,))

        return x + jnp.tanh(ff_gate) * ff_out


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
    query_partner_lstm: bool = False

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
        # Unpack scan input based on enabled flags
        if self.cross_agent_attn and self.query_partner_lstm:
            obs_flat, dones, partner_embed, partner_lstm_h = x
        elif self.cross_agent_attn:
            obs_flat, dones, partner_embed = x
        elif self.query_partner_lstm:
            obs_flat, dones, partner_lstm_h = x
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

        # Query from own LSTM state (+ partner's actor h when enabled)
        if self.query_partner_lstm:
            partner_lstm_h = jnp.where(dones[:, np.newaxis], 0.0, partner_lstm_h)
            own_state = jnp.concatenate(
                [lstm_h, lstm_c, jax.lax.stop_gradient(partner_lstm_h)], axis=-1)
        else:
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

        # Save own spatial features before cross-attention (for partner to use next step)
        own_spatial = features_with_pos.reshape(batch_size, fh * fw, -1)

        if self.cross_agent_attn:
            attended_flat = CrossAgentAttention(
                embed_dim=m * cm,
                num_heads=m,
                head_dim=cm,
                name="cross_agent_attn",
            )(attended_flat, jax.lax.stop_gradient(partner_embed))

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
            return (new_h, new_c), (lstm_out, attn_map, jax.lax.stop_gradient(own_spatial))
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
    query_partner_lstm: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        # Unpack network input based on enabled flags
        if self.cross_agent_attn and self.query_partner_lstm:
            obs, dones, avail_actions, partner_embed_actor, partner_embed_critic, plh_actor, plh_critic = x
        elif self.cross_agent_attn:
            obs, dones, avail_actions, partner_embed_actor, partner_embed_critic = x
        elif self.query_partner_lstm:
            obs, dones, avail_actions, plh_actor, plh_critic = x
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
            query_partner_lstm=self.query_partner_lstm,
        )

        # Build scan input tuples
        def _scan_input(partner_embed=None, partner_lstm_h=None):
            parts = [obs, dones]
            if self.cross_agent_attn:
                parts.append(partner_embed)
            if self.query_partner_lstm:
                parts.append(partner_lstm_h)
            return tuple(parts)

        # Actor path — partner's actor h feeds the actor query
        _plh_a = plh_actor if self.query_partner_lstm else None
        if self.cross_agent_attn:
            actor_lstm_state, (actor_embed, attn_map, actor_own_embed) = JAImageScannedLSTM(
                **rnn_kwargs, name="actor_lstm",
            )(actor_lstm_state, _scan_input(partner_embed_actor, _plh_a))
        else:
            actor_lstm_state, (actor_embed, attn_map) = JAImageScannedLSTM(
                **rnn_kwargs, name="actor_lstm",
            )(actor_lstm_state, _scan_input(partner_lstm_h=_plh_a))

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

        # Critic path — partner's critic h feeds the critic query
        _plh_c = plh_critic if self.query_partner_lstm else None
        if self.cross_agent_attn:
            critic_lstm_state, (critic_embed, _, critic_own_embed) = JAImageScannedLSTM(
                **rnn_kwargs, name="critic_lstm",
            )(critic_lstm_state, _scan_input(partner_embed_critic, _plh_c))
        else:
            critic_lstm_state, (critic_embed, _) = JAImageScannedLSTM(
                **rnn_kwargs, name="critic_lstm",
            )(critic_lstm_state, _scan_input(partner_lstm_h=_plh_c))

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
