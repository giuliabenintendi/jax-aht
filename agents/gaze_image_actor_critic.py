"""Gaze-based image actor-critic with a Gaussian foveal attention module.

This is an adaptation of "Gaze on the Prize" for the existing JaxAHT image
pipeline. The core design is:

  obs -> ResNet encoder -> feature map
      -> Gaussian gaze head -> spatial attention map
      -> foveated feature map -> FC -> LSTM -> actor / critic heads

The module also exposes a contrastive-feature method that reuses the same gaze
parameters on detached visual features for return-guided triplet learning.
"""
import functools
import numpy as np
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp

from agents.action_masking import mask_action_logits
from agents.ja_image_actor_critic import _compute_resnet_output_dims
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
    """Construct a normalized 2D Gaussian attention map on a feature grid.

    Args:
        feat_h: feature-map height
        feat_w: feature-map width
        mu_x, mu_y: center coordinates in [-1, 1]
        sigma_x, sigma_y: positive spread parameters in normalized coordinates
        rho: correlation term in [-1, 1]

    Returns:
        Attention map of shape (..., feat_h, feat_w), normalized over the last
        two axes.
    """
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
    inv_norm = 1.0 / jnp.maximum(2.0 * (1.0 - rho ** 2), 1e-6)
    mahal = (
        (dx / jnp.maximum(sigma_x, 1e-6)) ** 2
        + (dy / jnp.maximum(sigma_y, 1e-6)) ** 2
        - 2.0 * rho * dx * dy / jnp.maximum(sigma_x * sigma_y, 1e-6)
    )
    attn_logits = -inv_norm * mahal
    attn = jnp.exp(attn_logits - jnp.max(attn_logits, axis=(-2, -1), keepdims=True))
    return attn / jnp.maximum(attn.sum(axis=(-2, -1), keepdims=True), 1e-8)


class GaussianGazeHead(nn.Module):
    """Predict a normalized 2D Gaussian attention map over CNN features."""
    feat_h: int
    feat_w: int
    hidden_dim: int = 64
    min_sigma: float = 0.15
    max_rho: float = 0.95
    target_sigma_x: float = 0.45
    target_sigma_y: float = 0.45

    def setup(self):
        ys = jnp.linspace(-1.0, 1.0, self.feat_h)
        xs = jnp.linspace(-1.0, 1.0, self.feat_w)
        grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
        self.grid_x = grid_x
        self.grid_y = grid_y

    @nn.compact
    def __call__(self, feature_maps: jnp.ndarray):
        pooled = feature_maps.mean(axis=(1, 2))

        hidden = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="gaze_fc1",
        )(pooled)
        hidden = nn.tanh(hidden)
        raw = nn.Dense(
            5,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="gaze_params",
        )(hidden)

        mu_x = jnp.tanh(raw[:, 0])
        mu_y = jnp.tanh(raw[:, 1])
        sigma_x = nn.softplus(raw[:, 2]) + self.min_sigma
        sigma_y = nn.softplus(raw[:, 3]) + self.min_sigma
        rho = self.max_rho * jnp.tanh(raw[:, 4])
        attn = gaussian_attention_from_params(
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
        return attn, spread_loss, gaze_stats


class GazeImageScannedLSTM(nn.Module):
    """Scanned module: CNN -> Gaussian gaze -> foveation -> FC -> LSTM."""
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
    gaze_hidden_dim: int = 64
    contrastive_dim: int = 128

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

        zero_h = jnp.zeros((batch_size, self.lstm_hidden_dim), dtype=obs_flat.dtype)
        zero_c = jnp.zeros((batch_size, self.lstm_hidden_dim), dtype=obs_flat.dtype)
        lstm_h = jnp.where(dones[:, None], zero_h, lstm_h)
        lstm_c = jnp.where(dones[:, None], zero_c, lstm_c)

        image = obs_flat[:, :self._img_flat_dim].reshape(
            obs_flat.shape[0], self.img_height, self.img_width, self.num_channels
        )
        feature_maps = ResNetEncoder(
            num_blocks=self.conv_num_blocks,
            filters=self.conv_filters,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
            name="resnet_encoder",
        )(image)
        feature_embed = feature_maps.mean(axis=(1, 2))

        attn_map, spread_loss, gaze_stats = GaussianGazeHead(
            feat_h=self.feat_h,
            feat_w=self.feat_w,
            hidden_dim=self.gaze_hidden_dim,
            name="gaze_head",
        )(feature_maps)

        weighted = feature_maps * attn_map[..., None]
        weighted_flat = weighted.reshape(weighted.shape[0], -1)
        contrastive_repr = nn.Dense(
            self.contrastive_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="contrastive_proj",
        )(weighted_flat)
        contrastive_repr = nn.tanh(contrastive_repr)

        lstm_input = nn.relu(nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="input_fc1",
        )(weighted_flat))
        lstm_input = nn.relu(nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="input_fc2",
        )(lstm_input))
        new_carry, lstm_out = nn.OptimizedLSTMCell(
            features=self.lstm_hidden_dim,
            name="shared_lstm",
        )((lstm_h, lstm_c), lstm_input)

        aux = {
            "attn_map": attn_map,
            "feature_embed": feature_embed,
            "contrastive_repr": contrastive_repr,
            "spread_loss": spread_loss,
            "gaze_mu_x": gaze_stats["mu_x"],
            "gaze_mu_y": gaze_stats["mu_y"],
            "gaze_sigma_x": gaze_stats["sigma_x"],
            "gaze_sigma_y": gaze_stats["sigma_y"],
        }
        return new_carry, (lstm_out, aux)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return (
            jnp.zeros((batch_size, hidden_size)),
            jnp.zeros((batch_size, hidden_size)),
        )


class GazeImageActorCritic(nn.Module):
    """Shared-trunk image actor-critic with Gaussian gaze."""
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
    gaze_hidden_dim: int = 64
    contrastive_dim: int = 128

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
        (new_h, new_c), (lstm_out, aux) = GazeImageScannedLSTM(
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
            gaze_hidden_dim=self.gaze_hidden_dim,
            contrastive_dim=self.contrastive_dim,
            name="gaze_lstm",
        )(hidden, (obs, dones))

        actor_out = nn.relu(self.actor_fc1(lstm_out))
        actor_out = nn.relu(self.actor_fc2(actor_out))
        action_logits = self.actor_proj(actor_out)
        action_logits = mask_action_logits(action_logits, avail_actions)
        pi = distrax.Categorical(logits=action_logits)

        critic_out = nn.relu(self.critic_fc1(lstm_out))
        critic_out = nn.relu(self.critic_fc2(critic_out))
        value = self.critic_proj(critic_out)

        return (new_h, new_c), pi, jnp.squeeze(value, axis=-1), aux

    def contrastive_features(self, obs: jnp.ndarray):
        image = obs[:, :self._img_flat_dim].reshape(
            obs.shape[0], self.img_height, self.img_width, self.num_channels
        )
        feature_maps = ResNetEncoder(
            num_blocks=self.conv_num_blocks,
            filters=self.conv_filters,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
            name="resnet_encoder",
        )(image)
        feature_maps = jax.lax.stop_gradient(feature_maps)
        attn_map, spread_loss, gaze_stats = GaussianGazeHead(
            feat_h=self.feat_h,
            feat_w=self.feat_w,
            hidden_dim=self.gaze_hidden_dim,
            name="gaze_head",
        )(feature_maps)
        weighted = feature_maps * attn_map[..., None]
        weighted_flat = weighted.reshape(weighted.shape[0], -1)
        contrastive_repr = nn.Dense(
            self.contrastive_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="contrastive_proj",
        )(weighted_flat)
        contrastive_repr = nn.tanh(contrastive_repr)
        return {
            "contrastive_repr": contrastive_repr,
            "attn_map": attn_map,
            "spread_loss": spread_loss,
            "gaze_mu_x": gaze_stats["mu_x"],
            "gaze_mu_y": gaze_stats["mu_y"],
            "gaze_sigma_x": gaze_stats["sigma_x"],
            "gaze_sigma_y": gaze_stats["sigma_y"],
        }
