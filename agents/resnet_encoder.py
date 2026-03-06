"""ResNet CNN encoder following Forkel et al. 2025.

Residual blocks with projection shortcut for stride/channel mismatch.
Used by both JA-IPPO (without flatten, for spatial attention) and
Image-IPPO (with flatten, for FC input).
"""
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class ResidualBlock(nn.Module):
    channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.Conv(
            self.channels,
            (self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding=self.padding,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        y = nn.relu(y)
        y = nn.Conv(
            self.channels,
            (self.kernel_size, self.kernel_size),
            strides=1,
            padding=self.padding,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(y)

        # Projection shortcut for stride or channel mismatch
        shortcut = x
        if self.stride != 1 or x.shape[-1] != self.channels:
            shortcut = nn.Conv(
                self.channels,
                kernel_size=(1, 1),
                strides=self.stride,
                padding="VALID",
                kernel_init=orthogonal(1.0),
                bias_init=constant(0.0),
            )(x)

        return nn.relu(y + shortcut)


class ResNetEncoder(nn.Module):
    """ResNet CNN encoder. Returns spatial feature map (no flatten).

    Architecture:
      - Initial Conv(kernel, stride) + ReLU
      - N residual blocks (first block uses stride, rest use stride=1)
      - Output: (batch, H_out, W_out, filters)
    """
    num_blocks: int = 4
    filters: int = 32
    kernel_size: int = 3
    stride: int = 2
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.Conv(
            self.filters,
            (self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding=self.padding,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        y = nn.relu(y)

        for i in range(self.num_blocks):
            block_stride = self.stride if i == 0 else 1
            y = ResidualBlock(
                channels=self.filters,
                kernel_size=self.kernel_size,
                stride=block_stride,
                padding=self.padding,
            )(y)

        return y
