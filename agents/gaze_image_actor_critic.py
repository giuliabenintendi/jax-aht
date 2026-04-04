"""Paper-faithful Mott attention actor-critic for image observations.

This module follows the architecture described in:
  Mott et al., "Towards Interpretable Reinforcement Learning Using
  Attention Augmented Agents" (NeurIPS 2019)

and closely mirrors the local replication in:
  external/mott_attention_replication/attention.py

Paper-faithful components:
  - 2-layer vision CNN: Conv(8x8, stride 4, 32), Conv(4x4, stride 2, 64)
  - visual ConvLSTM core with 128 channels
  - direct split of visual output into K(8) and V(120)
  - cosine spatial basis with 64 channels appended to both K and V
  - top-down query network from previous policy-core output only
  - answer processor: 1026 -> 512 -> 256
  - policy core: LSTM(256)
  - post-core hidden layer: 256 -> 128 -> policy/value heads

Adaptation for this codebase:
  - keeps PPO/action-masking integration
  - keeps scalar value output instead of the replication's per-action value head
"""
import functools

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np

from agents.action_masking import mask_action_logits


VISION_CONV1_KERNEL = 8
VISION_CONV1_STRIDE = 4
VISION_CONV1_PADDING = 1
VISION_CONV1_CHANNELS = 32
VISION_CONV2_KERNEL = 4
VISION_CONV2_STRIDE = 2
VISION_CONV2_PADDING = 2
VISION_CONV2_CHANNELS = 64
VISION_LSTM_KERNEL = 3
VISION_LSTM_CHANNELS = 128
POST_CORE_HIDDEN_DIM = 128


def _conv_out_dim(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


def _compute_mott_output_dims(h: int, w: int) -> tuple[int, int]:
    h = _conv_out_dim(h, VISION_CONV1_KERNEL, VISION_CONV1_STRIDE, VISION_CONV1_PADDING)
    w = _conv_out_dim(w, VISION_CONV1_KERNEL, VISION_CONV1_STRIDE, VISION_CONV1_PADDING)
    h = _conv_out_dim(h, VISION_CONV2_KERNEL, VISION_CONV2_STRIDE, VISION_CONV2_PADDING)
    w = _conv_out_dim(w, VISION_CONV2_KERNEL, VISION_CONV2_STRIDE, VISION_CONV2_PADDING)
    return h, w


def make_mott_spatial_basis(height: int, width: int, channels: int = 64) -> jnp.ndarray:
    """Cosine spatial basis matching the replication's construction."""
    u = v = int(np.sqrt(channels))
    if u * v != channels:
        raise ValueError(f"channels={channels} must be a perfect square")

    p_h = jnp.arange(1, height + 1, dtype=jnp.float32)[:, None] * (jnp.pi / height)
    p_h = p_h * jnp.ones((1, width), dtype=jnp.float32)
    p_w = jnp.ones((height, 1), dtype=jnp.float32) * (
        jnp.arange(1, width + 1, dtype=jnp.float32)[None, :] * (jnp.pi / width)
    )

    u_basis = jnp.arange(1, u + 1, dtype=jnp.float32)[None, :]
    v_basis = jnp.arange(1, v + 1, dtype=jnp.float32)[None, :]
    a = p_h[..., None] * u_basis
    b = p_w[..., None] * v_basis
    return jnp.einsum("hwu,hwv->hwuv", jnp.cos(a), jnp.cos(b)).reshape(height, width, channels)


class MottConvLSTMCell(nn.Module):
    """ConvLSTM with peephole connections, matching the replication."""

    hidden_channels: int = VISION_LSTM_CHANNELS
    kernel_size: int = VISION_LSTM_KERNEL

    @nn.compact
    def __call__(self, carry, x):
        h, c = carry
        _, height, width, _ = x.shape
        padding = "SAME"

        wci = self.param("Wci", constant(0.0), (1, height, width, self.hidden_channels))
        wcf = self.param("Wcf", constant(0.0), (1, height, width, self.hidden_channels))
        wco = self.param("Wco", constant(0.0), (1, height, width, self.hidden_channels))

        conv_args = dict(
            features=self.hidden_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(1, 1),
            padding=padding,
        )

        i = jax.nn.sigmoid(
            nn.Conv(
                **conv_args,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="Wxi",
            )(x)
            + nn.Conv(
                **conv_args,
                kernel_init=orthogonal(1.0),
                use_bias=False,
                name="Whi",
            )(h)
            + c * wci
        )
        f = jax.nn.sigmoid(
            nn.Conv(
                **conv_args,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="Wxf",
            )(x)
            + nn.Conv(
                **conv_args,
                kernel_init=orthogonal(1.0),
                use_bias=False,
                name="Whf",
            )(h)
            + c * wcf
        )
        g = jnp.tanh(
            nn.Conv(
                **conv_args,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="Wxc",
            )(x)
            + nn.Conv(
                **conv_args,
                kernel_init=orthogonal(1.0),
                use_bias=False,
                name="Whc",
            )(h)
        )
        new_c = f * c + i * g
        o = jax.nn.sigmoid(
            nn.Conv(
                **conv_args,
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
                name="Wxo",
            )(x)
            + nn.Conv(
                **conv_args,
                kernel_init=orthogonal(1.0),
                use_bias=False,
                name="Who",
            )(h)
            + new_c * wco
        )
        new_h = o * jnp.tanh(new_c)
        return (new_h, new_c), new_h


class MottVisionNetwork(nn.Module):
    """Exact paper vision stack: 2 convs + ConvLSTM."""

    @nn.compact
    def __call__(self, carry, image):
        x = nn.Conv(
            features=VISION_CONV1_CHANNELS,
            kernel_size=(VISION_CONV1_KERNEL, VISION_CONV1_KERNEL),
            strides=(VISION_CONV1_STRIDE, VISION_CONV1_STRIDE),
            padding=[(VISION_CONV1_PADDING, VISION_CONV1_PADDING), (VISION_CONV1_PADDING, VISION_CONV1_PADDING)],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="vision_conv1",
        )(image)
        x = nn.relu(x)
        x = nn.Conv(
            features=VISION_CONV2_CHANNELS,
            kernel_size=(VISION_CONV2_KERNEL, VISION_CONV2_KERNEL),
            strides=(VISION_CONV2_STRIDE, VISION_CONV2_STRIDE),
            padding=[(VISION_CONV2_PADDING, VISION_CONV2_PADDING), (VISION_CONV2_PADDING, VISION_CONV2_PADDING)],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="vision_conv2",
        )(x)
        x = nn.relu(x)
        new_carry, output = MottConvLSTMCell(name="vision_lstm")(carry, x)
        return new_carry, output


class MottQueryNetwork(nn.Module):
    """Query MLP: 256 -> 128 -> 288 -> 288 reshaped to (4, 72)."""

    num_queries: int = 4
    query_dim: int = 72
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
    """Paper-style attention over split keys/values with cosine spatial basis."""

    feat_h: int
    feat_w: int
    key_dim: int = 8
    value_dim: int = 120
    num_queries: int = 4
    spatial_basis_depth: int = 64
    query_hidden_dim_1: int = 128
    query_hidden_dim_2: int = 288

    def setup(self):
        self.spatial_basis = make_mott_spatial_basis(
            self.feat_h, self.feat_w, self.spatial_basis_depth
        )

    @nn.compact
    def __call__(self, vision_output, prev_output, prev_reward, prev_action):
        batch_size = vision_output.shape[0]
        spatial = jnp.broadcast_to(
            self.spatial_basis[None, ...],
            (batch_size, self.feat_h, self.feat_w, self.spatial_basis_depth),
        )

        keys_raw, values_raw = jnp.split(
            vision_output, [self.key_dim], axis=-1
        )
        keys = jnp.concatenate([keys_raw, spatial], axis=-1)
        values = jnp.concatenate([values_raw, spatial], axis=-1)

        query_dim = self.key_dim + self.spatial_basis_depth
        queries = MottQueryNetwork(
            num_queries=self.num_queries,
            query_dim=query_dim,
            hidden_dim_1=self.query_hidden_dim_1,
            hidden_dim_2=self.query_hidden_dim_2,
            name="query_network",
        )(prev_output)

        attn_logits = jnp.einsum("bhwc,bqc->bhwq", keys, queries)
        attn_maps = jax.nn.softmax(attn_logits.reshape(batch_size, -1, self.num_queries), axis=1)
        attn_maps = attn_maps.reshape(batch_size, self.feat_h, self.feat_w, self.num_queries)
        answers = jnp.einsum("bhwq,bhwd->bqd", attn_maps, values)

        attn_maps_heads = jnp.transpose(attn_maps, (0, 3, 1, 2))
        mean_attn_map = attn_maps_heads.mean(axis=1)
        control_input = jnp.concatenate(
            [
                answers.reshape(batch_size, -1),
                queries.reshape(batch_size, -1),
                prev_reward[:, None].astype(vision_output.dtype),
                prev_action[:, None].astype(vision_output.dtype),
            ],
            axis=-1,
        )
        aux = {
            "attn_map": mean_attn_map,
            "attn_maps": attn_maps_heads,
            "answers": answers,
            "queries": queries,
        }
        return control_input, aux


class GazeImageScannedLSTM(nn.Module):
    """Paper-faithful Mott trunk scanned over time."""

    img_height: int
    img_width: int
    num_channels: int = 3
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
        self.feat_h, self.feat_w = _compute_mott_output_dims(self.img_height, self.img_width)
        self.vision = MottVisionNetwork(name="vision")
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
        self.policy_core = nn.OptimizedLSTMCell(
            features=self.lstm_hidden_dim,
            name="policy_core",
        )

    def initialize_carry(self, batch_size):
        vision_shape = (batch_size, self.feat_h, self.feat_w, VISION_LSTM_CHANNELS)
        return (
            jnp.zeros(vision_shape),
            jnp.zeros(vision_shape),
            jnp.zeros((batch_size, self.lstm_hidden_dim)),
            jnp.zeros((batch_size, self.lstm_hidden_dim)),
        )

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(self, carry, x):
        vis_h, vis_c, pol_h, pol_c = carry
        obs_flat, dones, prev_reward, prev_action = x
        batch_size = obs_flat.shape[0]

        zero_vis_h, zero_vis_c, zero_pol_h, zero_pol_c = self.initialize_carry(batch_size)
        done_mask_img = dones[:, None, None, None]
        vis_h = jnp.where(done_mask_img, zero_vis_h, vis_h)
        vis_c = jnp.where(done_mask_img, zero_vis_c, vis_c)
        pol_h = jnp.where(dones[:, None], zero_pol_h, pol_h)
        pol_c = jnp.where(dones[:, None], zero_pol_c, pol_c)
        prev_reward = jnp.where(dones, 0.0, prev_reward)
        prev_action = jnp.where(dones, 0.0, prev_action)

        image = obs_flat[:, :self._img_flat_dim].reshape(
            batch_size, self.img_height, self.img_width, self.num_channels
        )
        (new_vis_h, new_vis_c), vision_output = self.vision((vis_h, vis_c), image)

        control_input, aux = self.attention_readout(
            vision_output=vision_output,
            prev_output=pol_h,
            prev_reward=prev_reward,
            prev_action=prev_action,
        )

        answer = self.answer_fc1(control_input)
        answer = nn.relu(answer)
        answer = self.answer_fc2(answer)

        (new_pol_h, new_pol_c), policy_out = self.policy_core((pol_h, pol_c), answer)
        aux["answer"] = answer
        return (new_vis_h, new_vis_c, new_pol_h, new_pol_c), (policy_out, aux)


class GazeImageActorCritic(nn.Module):
    """Paper-faithful Mott actor-critic with shared visual and policy cores."""

    action_dim: int
    img_height: int
    img_width: int
    num_channels: int = 3
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
        self.output_fc = nn.Dense(
            POST_CORE_HIDDEN_DIM,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="output_fc",
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

        hidden, (policy_core_out, aux) = self.shared_trunk(
            hidden, (obs, dones, prev_reward, prev_action)
        )
        output = self.output_fc(policy_core_out)
        output = nn.relu(output)

        action_logits = self.policy_head(output)
        action_logits = mask_action_logits(action_logits, avail_actions)
        pi = distrax.Categorical(logits=action_logits)

        value = self.value_head(output)
        return hidden, pi, jnp.squeeze(value, axis=-1), aux
