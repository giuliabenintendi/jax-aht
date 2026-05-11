"""Visualize how card-game obs is partitioned by different attention-source
grids, and demonstrate the bilinear-interpolation upsampling used to overlay
attention heatmaps on the image (a la Abnar's attention-rollout tutorials).

Card-game obs is (21, 35, 3). Grids tested:
  - patches:    3 x 5  = 15 tokens (each patch = one tile, 7x7 px)
  - CNN cells:  6 x 9  = 54 tokens (current JA arch, stride=2 SAME, 2 downs)

Run:
    uv run python -m evaluation.patch_test
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs import make_env


def img_to_patches(img: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """Split image (H, W, C) into non-overlapping patches.

    Returns shape (N_patches, patch_size, patch_size, C) in row-major order.
    Raises if H or W not divisible by patch_size.
    """
    H, W, C = img.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"img shape ({H}, {W}) not divisible by patch_size={patch_size}")
    n_h = H // patch_size
    n_w = W // patch_size
    x = img.reshape(n_h, patch_size, n_w, patch_size, C)
    x = jnp.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(n_h * n_w, patch_size, patch_size, C)
    return x


def img_to_patches_flat(img: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """Same as `img_to_patches` but flattens each patch to a feature vector.

    Returns shape (N_patches, patch_size*patch_size*C).
    """
    patches = img_to_patches(img, patch_size)
    return patches.reshape(patches.shape[0], -1)


def attn_heatmap(attn_weights: jnp.ndarray, grid_h: int, grid_w: int,
                  target_h: int, target_w: int) -> jnp.ndarray:
    """JAX equivalent of `F.interpolate(..., mode='bilinear')` for attention.

    Args:
        attn_weights: (N,) flat attention vector over grid_h*grid_w tokens,
            in row-major order. Should be non-negative; renormalization is
            handled by the caller if a probability is desired.
        grid_h, grid_w: token grid dimensions.
        target_h, target_w: output spatial size (typically the input image).

    Returns:
        (target_h, target_w) heatmap, bilinearly upsampled.
    """
    grid = attn_weights.reshape(grid_h, grid_w)
    return jax.image.resize(grid, (target_h, target_w), method="bilinear")


def _draw_grid_lines(ax, n_h: int, n_w: int, img_h: int, img_w: int,
                     color: str = "white", lw: float = 0.6) -> None:
    """Draw a uniform grid of n_h x n_w cells over an image axis."""
    for r in range(1, n_h):
        y = r * (img_h / n_h) - 0.5
        ax.axhline(y=y, color=color, lw=lw, alpha=0.8)
    for c in range(1, n_w):
        x = c * (img_w / n_w) - 0.5
        ax.axvline(x=x, color=color, lw=lw, alpha=0.8)


def main():
    env_kwargs = dict(
        max_steps=8,
        obs_type="image",
        shuffle=True,
        other_play_position_shuffle=True,
        other_play_recolouring=True,
        communication=True,
    )
    env = make_env("card-game", env_kwargs)

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    img_flat = obs["agent_0"]
    img = img_flat.reshape(21, 35, 3)
    img_np = np.asarray(img)

    H, W = img.shape[:2]
    print(f"obs flat dim: {img_flat.shape}")
    print(f"img shape: {img.shape}, range [{img.min():.3f}, {img.max():.3f}]")

    # --- (a) ViT-style 7x7 patches: 3 x 5 = 15 tokens ---
    patches = img_to_patches(img, patch_size=7)
    patches_flat = img_to_patches_flat(img, patch_size=7)
    print(f"\npatches:    shape={patches.shape}, flat={patches_flat.shape}")

    # --- (b) Current CNN feature grid: 6 x 9 = 54 cells ---
    # conv_stride=2, SAME padding, 2 downsampling steps -> ceil(ceil(H/2)/2)
    feat_h = int(np.ceil(np.ceil(H / 2) / 2))   # 6
    feat_w = int(np.ceil(np.ceil(W / 2) / 2))   # 9
    print(f"CNN cells:  shape=({feat_h}, {feat_w})  ({feat_h * feat_w} cells total)")

    # --- (c) Synthetic attention heatmap demo ---
    # Pretend a trained model put most attention on card_2 (center card).
    # In the patch grid that's index 7 (row=1, col=2). For the CNN grid it
    # spans roughly cells (2-3, 4-5).
    fake_patch_attn = np.zeros(15, dtype=np.float32)
    fake_patch_attn[7] = 1.0  # all attention on the center card patch
    fake_patch_attn = fake_patch_attn / fake_patch_attn.sum()

    heatmap_from_patches = np.asarray(
        attn_heatmap(jnp.asarray(fake_patch_attn), 3, 5, H, W)
    )

    # And what the same "I'm looking at card 2" idea looks like at CNN resolution:
    fake_cnn_attn = np.zeros((feat_h, feat_w), dtype=np.float32)
    fake_cnn_attn[2:4, 4:6] = 1.0  # 2x2 block of cells covering center card
    fake_cnn_attn = fake_cnn_attn / fake_cnn_attn.sum()
    heatmap_from_cnn = np.asarray(
        attn_heatmap(jnp.asarray(fake_cnn_attn.flatten()), feat_h, feat_w, H, W)
    )

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))

    # Row 0: input partitioning grids overlaid on obs
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f"raw obs ({H}x{W})", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_np)
    _draw_grid_lines(axes[0, 1], 3, 5, H, W, color="cyan", lw=0.8)
    axes[0, 1].set_title("patch grid (3x5 = 15 tokens)", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_np)
    _draw_grid_lines(axes[0, 2], feat_h, feat_w, H, W, color="yellow", lw=0.6)
    axes[0, 2].set_title(f"CNN feature grid ({feat_h}x{feat_w} = {feat_h*feat_w} cells)", fontsize=10)
    axes[0, 2].axis("off")

    # Row 1: attention heatmap demo for both grids
    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title("(synthetic 'attend to card_2' attn)", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img_np)
    axes[1, 1].imshow(heatmap_from_patches, cmap="hot", alpha=0.55,
                      extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    _draw_grid_lines(axes[1, 1], 3, 5, H, W, color="cyan", lw=0.4)
    axes[1, 1].set_title("attn from patch grid -> bilinear upsample", fontsize=10)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(img_np)
    axes[1, 2].imshow(heatmap_from_cnn, cmap="hot", alpha=0.55,
                      extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    _draw_grid_lines(axes[1, 2], feat_h, feat_w, H, W, color="yellow", lw=0.4)
    axes[1, 2].set_title("attn from CNN grid -> bilinear upsample", fontsize=10)
    axes[1, 2].axis("off")

    fig.suptitle("Card-game obs: input partitioning + attention upsample demo "
                 "(synthetic 'attend to center card' attention)", fontsize=11)
    fig.tight_layout()
    out_path = "/tmp/patch_test.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved visualisation: {out_path}")


if __name__ == "__main__":
    main()
