"""Gaussian-gaze actor-critic for image observations.

Per-agent architecture:
  obs (flat) -> unpack image (H_px, W_px, num_channels)
  Image -> ResNet encoder -> features F  (H_out, W_out, filters)
  F -> Gaussian gaze head -> attention map A
  F * A -> foveated features -> FC -> FC -> LSTM -> projection

The actor and critic use separate recurrent trunks, matching the existing JA
image policy structure. The actor trunk also exposes a contrastive feature path
under the same parameter scope, so auxiliary triplet learning reuses the same
visual and gaze parameters that were initialized during the forward pass.
"""
import functools

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np

from agents.action_masking import mask_action_logits
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from agents.ja_utils import make_sinusoidal_spatial_basis
from agents.resnet_encoder import ResNetEncoder


def gaussian_attention_from_params(
    feat_h: int,
    feat_w: int,
    mu_x: jnp.ndarray,
    mu_y: jnp.ndarray,
    sigma_x: jnp.ndarray,
    sigma_y: jnp.ndarray,
    rho: jnp.ndarray,
):
    """Construct a normalized correlated 2D Gaussian over the feature grid."""
    ys = jnp.linspace(-1.0, 1.0, feat_h)
    xs = jnp.linspace(-1.0, 1.0, feat_w)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")

    mu_x = jnp.asarray(mu_x)[..., None, None]
    mu_y = jnp.asarray(mu_y)[..., None, None]
    sigma_x = jnp.asarray(sigma_x)[..., None, None]
    sigma_y = jnp.asarray(sigma_y)[..., None, None]
    rho = jnp.asarray(rho)[..., None, None]

    dx = grid_x - mu_x
    dy = grid_y - mu_y
    sigma_prod = jnp.maximum(sigma_x * sigma_y, 1e-6)
    inv_norm = 1.0 / jnp.maximum(2.0 * (1.0 - rho**2), 1e-6)
    mahal = (
        (dx / jnp.maximum(sigma_x, 1e-6)) ** 2
        + (dy / jnp.maximum(sigma_y, 1e-6)) ** 2
        - 2.0 * rho * dx * dy / sigma_prod
    )
    attn_logits = -inv_norm * mahal
    attn = jnp.exp(attn_logits - jnp.max(attn_logits, axis=(-2, -1), keepdims=True))
    return attn / jnp.maximum(attn.sum(axis=(-2, -1), keepdims=True), 1e-8)


class GaussianGazeHead(nn.Module):
    """Predict a normalized Gaussian attention map over CNN features."""

    feat_h: int
    feat_w: int
    spatial_basis_depth: int = 8
    conv_dim: int = 16
    min_sigma: float = 0.15
    max_rho: float = 0.95
    target_sigma_x: float = 0.20
    target_sigma_y: float = 0.20

    def setup(self):
        self.spatial_basis = make_sinusoidal_spatial_basis(
            self.feat_h, self.feat_w, self.spatial_basis_depth
        )

    @nn.compact
    def __call__(self, feature_maps: jnp.ndarray):
        batch_size = feature_maps.shape[0]
        spatial = jnp.broadcast_to(
            self.spatial_basis[None, ...],
            (batch_size, self.feat_h, self.feat_w, self.spatial_basis_depth),
        )
        gaze_inputs = jnp.concatenate([feature_maps, spatial], axis=-1)
        gaze_inputs = nn.Conv(
            features=self.conv_dim,
            kernel_size=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="gaze_conv1x1",
        )(gaze_inputs)
        gaze_inputs = nn.relu(gaze_inputs)
        spatial_logits = nn.Conv(
            features=1,
            kernel_size=(1, 1),
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="gaze_spatial_logits",
        )(gaze_inputs).squeeze(-1)
        spatial_probs = jax.nn.softmax(
            spatial_logits.reshape(batch_size, -1), axis=-1
        ).reshape(batch_size, self.feat_h, self.feat_w)

        ys = jnp.linspace(-1.0, 1.0, self.feat_h)
        xs = jnp.linspace(-1.0, 1.0, self.feat_w)
        grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x[None, ...]
        grid_y = grid_y[None, ...]

        mu_x = jnp.sum(spatial_probs * grid_x, axis=(1, 2))
        mu_y = jnp.sum(spatial_probs * grid_y, axis=(1, 2))

        dx = grid_x - mu_x[:, None, None]
        dy = grid_y - mu_y[:, None, None]
        var_x = jnp.sum(spatial_probs * dx ** 2, axis=(1, 2))
        var_y = jnp.sum(spatial_probs * dy ** 2, axis=(1, 2))
        cov_xy = jnp.sum(spatial_probs * dx * dy, axis=(1, 2))

        sigma_x = jnp.maximum(jnp.sqrt(jnp.maximum(var_x, 1e-6)), self.min_sigma)
        sigma_y = jnp.maximum(jnp.sqrt(jnp.maximum(var_y, 1e-6)), self.min_sigma)
        rho = jnp.clip(cov_xy / jnp.maximum(sigma_x * sigma_y, 1e-6), -self.max_rho, self.max_rho)

        attn_map = gaussian_attention_from_params(
            self.feat_h, self.feat_w, mu_x, mu_y, sigma_x, sigma_y, rho
        )
        spread_loss = (
            (jnp.log(sigma_x) - np.log(self.target_sigma_x)) ** 2
            + (jnp.log(sigma_y) - np.log(self.target_sigma_y)) ** 2
        )
        gaze_stats = {
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "rho": rho,
        }
        return attn_map, spread_loss, gaze_stats


class GazeImageScannedLSTM(nn.Module):
    """Scanned recurrent image trunk with Gaussian foveation."""

    img_height: int
    img_width: int
    num_channels: int = 3
    conv_filters: int = 32
    conv_num_blocks: int = 4
    conv_kernel_size: int = 3
    conv_stride: int = 2
    conv_padding: str = "SAME"
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
    gaze_spatial_basis_depth: int = 8
    gaze_conv_dim: int = 16
    contrastive_dim: int = 128
    target_sigma_x: float = 0.20
    target_sigma_y: float = 0.20

    def setup(self):
        self._img_flat_dim = self.img_height * self.img_width * self.num_channels
        self.feat_h, self.feat_w = _compute_resnet_output_dims(
            self.img_height,
            self.img_width,
            self.conv_stride,
            self.conv_kernel_size,
            self.conv_padding,
            self.conv_num_blocks,
        )
        self.resnet_encoder = ResNetEncoder(
            num_blocks=self.conv_num_blocks,
            filters=self.conv_filters,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
            name="resnet_encoder",
        )
        self.gaze_head = GaussianGazeHead(
            feat_h=self.feat_h,
            feat_w=self.feat_w,
            spatial_basis_depth=self.gaze_spatial_basis_depth,
            conv_dim=self.gaze_conv_dim,
            target_sigma_x=self.target_sigma_x,
            target_sigma_y=self.target_sigma_y,
            name="gaze_head",
        )
        self.contrastive_proj = nn.Dense(
            self.contrastive_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="contrastive_proj",
        )
        self.input_fc1 = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="input_fc1",
        )
        self.input_fc2 = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="input_fc2",
        )
        self.shared_lstm = nn.OptimizedLSTMCell(
            features=self.lstm_hidden_dim,
            name="shared_lstm",
        )

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return (
            jnp.zeros((batch_size, hidden_size)),
            jnp.zeros((batch_size, hidden_size)),
        )

    def _encode_gaze_features(self, obs_flat, stop_gradient=False):
        image = obs_flat[:, :self._img_flat_dim].reshape(
            obs_flat.shape[0], self.img_height, self.img_width, self.num_channels
        )
        feature_maps = self.resnet_encoder(image)
        if stop_gradient:
            feature_maps = jax.lax.stop_gradient(feature_maps)

        feature_embed = feature_maps.mean(axis=(1, 2))
        attn_map, spread_loss, gaze_stats = self.gaze_head(feature_maps)
        weighted = feature_maps * attn_map[..., None]
        weighted_pool = weighted.sum(axis=(1, 2))
        contrastive_repr = self.contrastive_proj(weighted_pool)
        contrastive_repr = nn.tanh(contrastive_repr)

        aux = {
            "attn_map": attn_map,
            "feature_embed": feature_embed,
            "contrastive_repr": contrastive_repr,
            "spread_loss": spread_loss,
            "gaze_mu_x": gaze_stats["mu_x"],
            "gaze_mu_y": gaze_stats["mu_y"],
            "gaze_sigma_x": gaze_stats["sigma_x"],
            "gaze_sigma_y": gaze_stats["sigma_y"],
            "gaze_rho": gaze_stats["rho"],
        }
        return weighted_pool, aux

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(self, carry, x):
        lstm_h, lstm_c = carry
        obs_flat, dones = x
        batch_size = obs_flat.shape[0]

        zero_h, zero_c = self.initialize_carry(batch_size, self.lstm_hidden_dim)
        lstm_h = jnp.where(dones[:, None], zero_h, lstm_h)
        lstm_c = jnp.where(dones[:, None], zero_c, lstm_c)

        weighted_flat, aux = self._encode_gaze_features(obs_flat, stop_gradient=False)

        lstm_input = self.input_fc1(weighted_flat)
        lstm_input = nn.relu(lstm_input)
        lstm_input = self.input_fc2(lstm_input)
        lstm_input = nn.relu(lstm_input)

        new_carry, lstm_out = self.shared_lstm((lstm_h, lstm_c), lstm_input)
        return new_carry, (lstm_out, aux)

    def encode_contrastive(self, obs_flat: jnp.ndarray):
        _, aux = self._encode_gaze_features(obs_flat, stop_gradient=True)
        return aux


class GazeImageActorCritic(nn.Module):
    """Actor-critic with Gaussian gaze and separate actor/critic trunks."""

    action_dim: int
    img_height: int
    img_width: int
    num_channels: int = 3
    conv_filters: int = 32
    conv_num_blocks: int = 4
    conv_kernel_size: int = 3
    conv_stride: int = 2
    conv_padding: str = "SAME"
    fc_hidden_dim: int = 64
    lstm_hidden_dim: int = 64
    gaze_spatial_basis_depth: int = 8
    gaze_conv_dim: int = 16
    contrastive_dim: int = 128
    target_sigma_x: float = 0.20
    target_sigma_y: float = 0.20

    def setup(self):
        trunk_kwargs = dict(
            img_height=self.img_height,
            img_width=self.img_width,
            num_channels=self.num_channels,
            conv_filters=self.conv_filters,
            conv_num_blocks=self.conv_num_blocks,
            conv_kernel_size=self.conv_kernel_size,
            conv_stride=self.conv_stride,
            conv_padding=self.conv_padding,
            fc_hidden_dim=self.fc_hidden_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            gaze_spatial_basis_depth=self.gaze_spatial_basis_depth,
            gaze_conv_dim=self.gaze_conv_dim,
            contrastive_dim=self.contrastive_dim,
            target_sigma_x=self.target_sigma_x,
            target_sigma_y=self.target_sigma_y,
        )
        self.actor_lstm = GazeImageScannedLSTM(**trunk_kwargs, name="actor_lstm")
        self.critic_lstm = GazeImageScannedLSTM(**trunk_kwargs, name="critic_lstm")
        self.actor_fc1 = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_fc1",
        )
        self.actor_fc2 = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_fc2",
        )
        self.actor_proj = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_proj",
        )
        self.critic_fc1 = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_fc1",
        )
        self.critic_fc2 = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_fc2",
        )
        self.critic_proj = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_proj",
        )

    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        actor_lstm_state, critic_lstm_state = hidden

        actor_lstm_state, (actor_embed, actor_aux) = self.actor_lstm(
            actor_lstm_state, (obs, dones)
        )
        actor_out = nn.relu(self.actor_fc1(actor_embed))
        actor_out = nn.relu(self.actor_fc2(actor_out))
        action_logits = self.actor_proj(actor_out)
        action_logits = mask_action_logits(action_logits, avail_actions)
        pi = distrax.Categorical(logits=action_logits)

        critic_lstm_state, (critic_embed, _) = self.critic_lstm(
            critic_lstm_state, (obs, dones)
        )
        critic_out = nn.relu(self.critic_fc1(critic_embed))
        critic_out = nn.relu(self.critic_fc2(critic_out))
        value = self.critic_proj(critic_out)

        new_hidden = (actor_lstm_state, critic_lstm_state)
        return new_hidden, pi, jnp.squeeze(value, axis=-1), actor_aux

    def contrastive_features(self, obs: jnp.ndarray):
        return self.actor_lstm.encode_contrastive(obs)
