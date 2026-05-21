"""Joint Attention actor-critic modules."""
import functools

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np

from agents.action_masking import mask_action_logits
from agents.ja_utils import make_sinusoidal_spatial_basis
from agents.resnet_encoder import ResNetEncoder


# Observation channel layout for symbolic Overcooked observations.
_POS_CHANNELS = slice(0, 2)
_EGO_DIR_CHANNELS = slice(2, 6)
_PARTNER_DIR_CHANNELS = slice(6, 10)
_IMAGE_CHANNELS = slice(10, 26)


def _compute_resnet_output_dims(h, w, stride, kernel_size, padding, num_blocks):
    """Compute spatial dims after the image JA ResNet encoder."""
    for _ in range(1 + min(1, num_blocks)):
        if padding == "SAME":
            h = int(np.ceil(h / stride))
            w = int(np.ceil(w / stride))
        else:
            h = (h - kernel_size) // stride + 1
            w = (w - kernel_size) // stride + 1
    return h, w


def _extract_symbolic_direction_features(obs_grid):
    ego_dir = obs_grid[..., _EGO_DIR_CHANNELS].sum(axis=(1, 2))
    partner_dir = obs_grid[..., _PARTNER_DIR_CHANNELS].sum(axis=(1, 2))
    return jnp.concatenate([ego_dir, partner_dir], axis=-1)


def _extract_symbolic_position_features(obs_grid, width):
    batch_size = obs_grid.shape[0]

    ego_pos_grid = obs_grid[..., 0]
    ego_flat = jnp.argmax(ego_pos_grid.reshape(batch_size, -1), axis=-1)
    ego_y = ego_flat // width
    ego_x = ego_flat % width

    partner_pos_grid = obs_grid[..., 1]
    partner_flat = jnp.argmax(partner_pos_grid.reshape(batch_size, -1), axis=-1)
    partner_y = partner_flat // width
    partner_x = partner_flat % width

    return jnp.stack(
        [
            ego_x.astype(jnp.float32),
            ego_y.astype(jnp.float32),
            partner_x.astype(jnp.float32),
            partner_y.astype(jnp.float32),
        ],
        axis=-1,
    )


class JAScannedLSTM(nn.Module):
    """Scanned JA core with a configurable symbolic or image encoder."""

    encoder_type: str = "symbolic"

    # Symbolic encoder config.
    obs_height: int | None = None
    obs_width: int | None = None
    obs_channels: int = 26

    # Image encoder config.
    img_height: int | None = None
    img_width: int | None = None
    num_channels: int = 3
    conv_num_blocks: int = 4
    conv_kernel_size: int = 3
    conv_stride: int = 2
    conv_padding: str = "SAME"
    message_dim: int = 0
    scalar_dim: int = 0
    query_partner_lstm: bool = False
    keep_attn_heads: bool = False

    # Shared JA config.
    conv_filters: int = 64
    num_heads: int = 4
    head_features: int = 16
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
    spatial_basis_depth: int = 8
    scalar_embed_dim: int = 5

    def setup(self):
        if self.encoder_type == "symbolic":
            if self.obs_height is None or self.obs_width is None:
                raise ValueError("symbolic JA requires obs_height and obs_width")
            self._spatial_h = self.obs_height
            self._spatial_w = self.obs_width
            self._img_flat_dim = None
        elif self.encoder_type == "image":
            if self.img_height is None or self.img_width is None:
                raise ValueError("image JA requires img_height and img_width")
            self._spatial_h, self._spatial_w = _compute_resnet_output_dims(
                self.img_height,
                self.img_width,
                self.conv_stride,
                self.conv_kernel_size,
                self.conv_padding,
                self.conv_num_blocks,
            )
            self._img_flat_dim = self.img_height * self.img_width * self.num_channels
        else:
            raise ValueError(f"Unsupported JA encoder_type: {self.encoder_type}")

        self.spatial_basis = make_sinusoidal_spatial_basis(
            self._spatial_h,
            self._spatial_w,
            self.spatial_basis_depth,
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
        lstm_h, lstm_c = carry
        if self.query_partner_lstm:
            obs_flat, dones, partner_lstm_h = x
        else:
            obs_flat, dones = x
            partner_lstm_h = None

        batch_size = obs_flat.shape[0]
        m, cm = self.num_heads, self.head_features

        zero_h, zero_c = self.initialize_carry(batch_size, self.lstm_hidden_dim)
        lstm_h = jnp.where(dones[:, None], zero_h, lstm_h)
        lstm_c = jnp.where(dones[:, None], zero_c, lstm_c)

        suffix_parts = []
        if self.encoder_type == "symbolic":
            obs_grid = obs_flat.reshape(
                batch_size,
                self.obs_height,
                self.obs_width,
                self.obs_channels,
            )
            spatial_input = jnp.concatenate(
                [obs_grid[..., _POS_CHANNELS], obs_grid[..., _IMAGE_CHANNELS]],
                axis=-1,
            )
            features = nn.Conv(
                features=self.conv_filters,
                kernel_size=(3, 3),
                padding="SAME",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="feature_conv",
            )(spatial_input)
            features = nn.relu(features)

            direction = _extract_symbolic_direction_features(obs_grid)
            suffix_parts.append(
                nn.Dense(
                    self.scalar_embed_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name="dir_embed",
                )(direction)
            )
            suffix_parts.append(
                nn.Dense(
                    self.scalar_embed_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name="pos_embed",
                )(_extract_symbolic_position_features(obs_grid, self.obs_width))
            )
        else:
            img_flat = obs_flat[:, : self._img_flat_dim]
            image = img_flat.reshape(
                batch_size,
                self.img_height,
                self.img_width,
                self.num_channels,
            )
            features = ResNetEncoder(
                num_blocks=self.conv_num_blocks,
                filters=self.conv_filters,
                kernel_size=self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.conv_padding,
                name="resnet_encoder",
            )(image)

            suffix_start = self._img_flat_dim
            if self.message_dim > 0:
                suffix_parts.append(
                    obs_flat[:, suffix_start : suffix_start + self.message_dim]
                )
                suffix_start += self.message_dim
            if self.scalar_dim > 0:
                scalar_input = obs_flat[:, suffix_start : suffix_start + self.scalar_dim]
                suffix_parts.append(
                    nn.Dense(
                        self.scalar_embed_dim,
                        kernel_init=orthogonal(np.sqrt(2)),
                        bias_init=constant(0.0),
                        name="scalar_embed",
                    )(scalar_input)
                )

        spatial = jnp.broadcast_to(
            self.spatial_basis[None, ...],
            (
                batch_size,
                self._spatial_h,
                self._spatial_w,
                self.spatial_basis_depth,
            ),
        )
        features_with_pos = jnp.concatenate([features, spatial], axis=-1)

        keys = nn.Conv(
            features=m * cm,
            kernel_size=(1, 1),
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="key_conv",
        )(features_with_pos).reshape(batch_size, self._spatial_h * self._spatial_w, m, cm)

        values = nn.Conv(
            features=m * cm,
            kernel_size=(1, 1),
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="value_conv",
        )(features_with_pos).reshape(batch_size, self._spatial_h * self._spatial_w, m, cm)

        if self.query_partner_lstm:
            partner_lstm_h = jnp.where(dones[:, None], 0.0, partner_lstm_h)
            own_state = jnp.concatenate(
                [lstm_h, lstm_c, jax.lax.stop_gradient(partner_lstm_h)],
                axis=-1,
            )
        else:
            own_state = jnp.concatenate([lstm_h, lstm_c], axis=-1)

        queries = nn.Dense(
            m * cm,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="query_ffn",
        )(own_state).reshape(batch_size, m, cm)

        attn_logits = jnp.einsum("bnmc,bmc->bnm", keys, queries)
        attn_weights = jax.nn.softmax(attn_logits, axis=1)

        attended = jnp.einsum("bnm,bnmc->bmc", attn_weights, values)
        attended_flat = attended.reshape(batch_size, m * cm)

        if self.keep_attn_heads:
            attn_map = attn_weights.reshape(batch_size, self._spatial_h, self._spatial_w, m)
        else:
            attn_map = attn_weights.mean(axis=-1).reshape(
                batch_size,
                self._spatial_h,
                self._spatial_w,
            )

        lstm_input = jnp.concatenate([attended_flat] + suffix_parts, axis=-1)
        lstm_input = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="input_fc1",
        )(lstm_input)
        lstm_input = nn.relu(lstm_input)
        lstm_input = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="input_fc2",
        )(lstm_input)
        lstm_input = nn.relu(lstm_input)

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


class JAActorCritic(nn.Module):
    """Unified JA actor-critic with configurable encoder and heads."""

    action_dim: int
    encoder_type: str = "symbolic"

    # Symbolic config.
    obs_height: int | None = None
    obs_width: int | None = None
    obs_channels: int = 26

    # Image config.
    img_height: int | None = None
    img_width: int | None = None
    num_channels: int = 3
    conv_num_blocks: int = 4
    conv_kernel_size: int = 3
    conv_stride: int = 2
    conv_padding: str = "SAME"
    message_dim: int = 0
    scalar_dim: int = 0
    query_partner_lstm: bool = False
    keep_attn_heads: bool = False

    # Shared config.
    conv_filters: int = 64
    num_heads: int = 4
    head_features: int = 16
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
    spatial_basis_depth: int = 8
    scalar_embed_dim: int = 5
    post_lstm_fc_layers: int = 0
    post_lstm_fc_hidden_dim: int | None = None
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        if self.query_partner_lstm:
            obs, dones, avail_actions, plh_actor, plh_critic = x
        else:
            obs, dones, avail_actions = x
            plh_actor = plh_critic = None

        actor_lstm_state, critic_lstm_state = hidden

        rnn_kwargs = dict(
            encoder_type=self.encoder_type,
            obs_height=self.obs_height,
            obs_width=self.obs_width,
            obs_channels=self.obs_channels,
            img_height=self.img_height,
            img_width=self.img_width,
            num_channels=self.num_channels,
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
            message_dim=self.message_dim,
            scalar_dim=self.scalar_dim,
            scalar_embed_dim=self.scalar_embed_dim,
            query_partner_lstm=self.query_partner_lstm,
            keep_attn_heads=self.keep_attn_heads,
        )

        def scan_input(partner_lstm_h=None):
            if self.query_partner_lstm:
                return obs, dones, partner_lstm_h
            return obs, dones

        actor_lstm_state, (actor_embed, attn_map) = JAScannedLSTM(
            **rnn_kwargs,
            name="actor_lstm",
        )(actor_lstm_state, scan_input(plh_actor))

        critic_lstm_state, (critic_embed, _) = JAScannedLSTM(
            **rnn_kwargs,
            name="critic_lstm",
        )(critic_lstm_state, scan_input(plh_critic))

        activation = nn.relu if self.activation == "relu" else nn.tanh
        post_lstm_hidden_dim = self.post_lstm_fc_hidden_dim or self.fc_hidden_dim

        def project_head(embed, prefix, output_dim, kernel_scale):
            out = embed
            for layer_idx in range(self.post_lstm_fc_layers):
                out = nn.Dense(
                    post_lstm_hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name=f"{prefix}_fc{layer_idx + 1}",
                )(out)
                out = activation(out)
            return nn.Dense(
                output_dim,
                kernel_init=orthogonal(kernel_scale),
                bias_init=constant(0.0),
                name=f"{prefix}_proj",
            )(out)

        action_logits = project_head(actor_embed, "actor", self.action_dim, 0.01)
        action_logits = mask_action_logits(action_logits, avail_actions)
        pi = distrax.Categorical(logits=action_logits)

        value = project_head(critic_embed, "critic", 1, 1.0)
        new_hidden = (actor_lstm_state, critic_lstm_state)
        return new_hidden, pi, jnp.squeeze(value, axis=-1), attn_map
