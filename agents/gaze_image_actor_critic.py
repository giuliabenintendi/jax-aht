"""Mott-style attention actor-critic for image observations.

This module ports the core attention-controller interface from:
  Mott et al., "Towards Interpretable Reinforcement Learning Using
  Attention Augmented Agents" (NeurIPS 2019)
and the accompanying replication in:
  external/mott_attention_replication/attention.py

Paper-faithful choices implemented here:
  - one shared recurrent controller for policy + value
  - queries generated from the previous recurrent output only
  - multiple top-down queries over spatial keys/values
  - controller input = answers + queries + previous reward + previous action

Adaptation for this codebase:
  - uses the existing ResNet visual encoder instead of the paper's Atari CNN
    + ConvLSTM visual stack
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


def attention_map_moments(attn_map: jnp.ndarray):
    """Summarize a normalized spatial attention map with first/second moments."""
    feat_h, feat_w = attn_map.shape[-2:]
    ys = jnp.linspace(-1.0, 1.0, feat_h)
    xs = jnp.linspace(-1.0, 1.0, feat_w)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x[None, ...]
    grid_y = grid_y[None, ...]

    mu_x = jnp.sum(attn_map * grid_x, axis=(1, 2))
    mu_y = jnp.sum(attn_map * grid_y, axis=(1, 2))

    dx = grid_x - mu_x[:, None, None]
    dy = grid_y - mu_y[:, None, None]
    var_x = jnp.sum(attn_map * dx ** 2, axis=(1, 2))
    var_y = jnp.sum(attn_map * dy ** 2, axis=(1, 2))
    cov_xy = jnp.sum(attn_map * dx * dy, axis=(1, 2))

    sigma_x = jnp.sqrt(jnp.maximum(var_x, 1e-6))
    sigma_y = jnp.sqrt(jnp.maximum(var_y, 1e-6))
    rho = jnp.clip(cov_xy / jnp.maximum(sigma_x * sigma_y, 1e-6), -0.95, 0.95)
    return {
        "mu_x": mu_x,
        "mu_y": mu_y,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "rho": rho,
    }


class MottQueryNetwork(nn.Module):
    """Top-down query MLP from previous recurrent output.

    Mirrors the replication's QueryNetwork structure:
      hidden -> 128 -> 288 -> num_queries * query_dim
    """

    num_queries: int
    query_dim: int
    hidden_dim_1: int = 128
    hidden_dim_2: int = 288

    @nn.compact
    def __call__(self, prev_output: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.hidden_dim_1,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="query_fc1",
        )(prev_output)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim_2,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="query_fc2",
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.num_queries * self.query_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="query_fc3",
        )(x)
        return x.reshape(prev_output.shape[0], self.num_queries, self.query_dim)


class MottAttentionReadout(nn.Module):
    """Mott-style multi-query readout over spatial keys and values."""

    feat_h: int
    feat_w: int
    key_dim: int = 8
    value_dim: int = 120
    num_queries: int = 4
    spatial_basis_depth: int = 64
    query_hidden_dim_1: int = 128
    query_hidden_dim_2: int = 288

    def setup(self):
        self.spatial_basis = make_sinusoidal_spatial_basis(
            self.feat_h, self.feat_w, self.spatial_basis_depth
        )

    @nn.compact
    def __call__(
        self,
        feature_maps: jnp.ndarray,
        prev_output: jnp.ndarray,
        prev_reward: jnp.ndarray,
        prev_action: jnp.ndarray,
    ):
        batch_size = feature_maps.shape[0]
        spatial = jnp.broadcast_to(
            self.spatial_basis[None, ...],
            (batch_size, self.feat_h, self.feat_w, self.spatial_basis_depth),
        )

        key_maps = nn.Conv(
            features=self.key_dim,
            kernel_size=(1, 1),
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="key_conv",
        )(feature_maps)
        value_maps = nn.Conv(
            features=self.value_dim,
            kernel_size=(1, 1),
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="value_conv",
        )(feature_maps)

        keys = jnp.concatenate([key_maps, spatial], axis=-1)
        values = jnp.concatenate([value_maps, spatial], axis=-1)

        query_dim = self.key_dim + self.spatial_basis_depth
        queries = MottQueryNetwork(
            num_queries=self.num_queries,
            query_dim=query_dim,
            hidden_dim_1=self.query_hidden_dim_1,
            hidden_dim_2=self.query_hidden_dim_2,
            name="query_network",
        )(prev_output)

        keys = keys.reshape(batch_size, self.feat_h * self.feat_w, query_dim)
        values = values.reshape(
            batch_size, self.feat_h * self.feat_w, self.value_dim + self.spatial_basis_depth
        )

        attn_logits = jnp.einsum("bnc,bqc->bnq", keys, queries) / np.sqrt(query_dim)
        attn_weights = jax.nn.softmax(attn_logits, axis=1)
        answers = jnp.einsum("bnq,bnd->bqd", attn_weights, values)

        attn_maps = jnp.transpose(attn_weights, (0, 2, 1)).reshape(
            batch_size, self.num_queries, self.feat_h, self.feat_w
        )
        mean_attn_map = attn_maps.mean(axis=1)
        gaze_stats = attention_map_moments(mean_attn_map)

        control_input = jnp.concatenate(
            [
                answers.reshape(batch_size, -1),
                queries.reshape(batch_size, -1),
                prev_reward[:, None].astype(feature_maps.dtype),
                prev_action[:, None].astype(feature_maps.dtype),
            ],
            axis=-1,
        )

        aux = {
            "attn_map": mean_attn_map,
            "attn_maps": attn_maps,
            "answers": answers,
            "queries": queries,
            "gaze_mu_x": gaze_stats["mu_x"],
            "gaze_mu_y": gaze_stats["mu_y"],
            "gaze_sigma_x": gaze_stats["sigma_x"],
            "gaze_sigma_y": gaze_stats["sigma_y"],
            "gaze_rho": gaze_stats["rho"],
        }
        return control_input, aux


class GazeImageScannedLSTM(nn.Module):
    """Shared Mott-style visual/controller trunk scanned over time."""

    img_height: int
    img_width: int
    num_channels: int = 3
    conv_filters: int = 32
    conv_num_blocks: int = 4
    conv_kernel_size: int = 3
    conv_stride: int = 2
    conv_padding: str = "SAME"
    fc_hidden_dim: int = 512
    lstm_hidden_dim: int = 256
    gaze_spatial_basis_depth: int = 64
    gaze_key_dim: int = 8
    gaze_value_dim: int = 120
    gaze_num_queries: int = 4
    query_hidden_dim_1: int = 128
    query_hidden_dim_2: int = 288

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
        self.attention_readout = MottAttentionReadout(
            feat_h=self.feat_h,
            feat_w=self.feat_w,
            key_dim=self.gaze_key_dim,
            value_dim=self.gaze_value_dim,
            num_queries=self.gaze_num_queries,
            spatial_basis_depth=self.gaze_spatial_basis_depth,
            query_hidden_dim_1=self.query_hidden_dim_1,
            query_hidden_dim_2=self.query_hidden_dim_2,
            name="attention_readout",
        )
        self.answer_fc1 = nn.Dense(
            self.fc_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="answer_fc1",
        )
        self.answer_fc2 = nn.Dense(
            self.lstm_hidden_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="answer_fc2",
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

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(self, carry, x):
        lstm_h, lstm_c = carry
        obs_flat, dones, prev_reward, prev_action = x
        batch_size = obs_flat.shape[0]

        zero_h, zero_c = self.initialize_carry(batch_size, self.lstm_hidden_dim)
        lstm_h = jnp.where(dones[:, None], zero_h, lstm_h)
        lstm_c = jnp.where(dones[:, None], zero_c, lstm_c)
        prev_reward = jnp.where(dones, 0.0, prev_reward)
        prev_action = jnp.where(dones, 0.0, prev_action)

        image = obs_flat[:, :self._img_flat_dim].reshape(
            batch_size, self.img_height, self.img_width, self.num_channels
        )
        feature_maps = self.resnet_encoder(image)

        control_input, aux = self.attention_readout(
            feature_maps,
            prev_output=lstm_h,
            prev_reward=prev_reward,
            prev_action=prev_action,
        )

        controller_input = self.answer_fc1(control_input)
        controller_input = nn.relu(controller_input)
        controller_input = self.answer_fc2(controller_input)

        new_carry, lstm_out = self.shared_lstm((lstm_h, lstm_c), controller_input)
        aux["controller_input"] = controller_input
        return new_carry, (lstm_out, aux)


class GazeImageActorCritic(nn.Module):
    """Shared-core Mott-style actor-critic with interpretable attention maps."""

    action_dim: int
    img_height: int
    img_width: int
    num_channels: int = 3
    conv_filters: int = 32
    conv_num_blocks: int = 4
    conv_kernel_size: int = 3
    conv_stride: int = 2
    conv_padding: str = "SAME"
    fc_hidden_dim: int = 512
    lstm_hidden_dim: int = 256
    gaze_spatial_basis_depth: int = 64
    gaze_key_dim: int = 8
    gaze_value_dim: int = 120
    gaze_num_queries: int = 4
    query_hidden_dim_1: int = 128
    query_hidden_dim_2: int = 288

    def setup(self):
        self.shared_trunk = GazeImageScannedLSTM(
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
            gaze_key_dim=self.gaze_key_dim,
            gaze_value_dim=self.gaze_value_dim,
            gaze_num_queries=self.gaze_num_queries,
            query_hidden_dim_1=self.query_hidden_dim_1,
            query_hidden_dim_2=self.query_hidden_dim_2,
            name="shared_trunk",
        )
        self.policy_head = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="policy_head",
        )
        self.value_head = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="value_head",
        )

    def __call__(self, hidden, x):
        if len(x) == 3:
            obs, dones, avail_actions = x
            prev_reward = jnp.zeros(obs.shape[:2], dtype=obs.dtype)
            prev_action = jnp.zeros(obs.shape[:2], dtype=obs.dtype)
        else:
            obs, dones, avail_actions, prev_reward, prev_action = x

        hidden, (shared_embed, aux) = self.shared_trunk(
            hidden, (obs, dones, prev_reward, prev_action)
        )

        action_logits = self.policy_head(shared_embed)
        action_logits = mask_action_logits(action_logits, avail_actions)
        pi = distrax.Categorical(logits=action_logits)

        value = self.value_head(shared_embed)
        return hidden, pi, jnp.squeeze(value, axis=-1), aux
