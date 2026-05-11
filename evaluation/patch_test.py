"""Quick test: split card-game observations into ViT-style patches.

Card-game obs is (21, 35, 3) with GRID_ROWS=3, GRID_COLS=5, TILE_PIXELS=7.
With patch_size=7 we get a 3x5 = 15-patch grid where each patch is exactly
one tile. Card tiles are patches 5-9 in row-major order.

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
    # (H, W, C) -> (n_h, p, n_w, p, C) -> (n_h, n_w, p, p, C) -> (n_h*n_w, p, p, C)
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

    print(f"obs flat dim: {img_flat.shape}")
    print(f"img shape: {img.shape}, range [{img.min():.3f}, {img.max():.3f}]")

    patches = img_to_patches(img, patch_size=7)
    print(f"patches shape: {patches.shape}  (15 patches of 7x7x3)")

    patches_flat = img_to_patches_flat(img, patch_size=7)
    print(f"patches_flat shape: {patches_flat.shape}  (15 patches of 147 features)")

    # Visualize: original image (left) + 3x5 patch grid (right)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(img_np)
    axes[0].set_title(f"agent_0 obs ({img.shape[0]}x{img.shape[1]})")
    axes[0].axis("off")
    for c in range(5):
        x_center = 7 * c + 3.5
        axes[0].axvline(x=7 * c - 0.5, color="white", lw=0.5, alpha=0.7)
    for r in range(3):
        axes[0].axhline(y=7 * r - 0.5, color="white", lw=0.5, alpha=0.7)

    grid = np.asarray(patches).reshape(3, 5, 7, 7, 3)
    # Reassemble with separators
    sep = 2
    grid_img = np.ones((3 * 7 + 4 * sep, 5 * 7 + 6 * sep, 3), dtype=np.float32)
    for r in range(3):
        for c in range(5):
            y0 = sep + r * (7 + sep)
            x0 = sep + c * (7 + sep)
            grid_img[y0 : y0 + 7, x0 : x0 + 7] = grid[r, c]
    axes[1].imshow(grid_img)
    axes[1].set_title("3x5 = 15 patches (patch_size=7)")
    axes[1].axis("off")
    for i, (r, c) in enumerate([(r, c) for r in range(3) for c in range(5)]):
        y_center = sep + r * (7 + sep) + 3.5
        x_center = sep + c * (7 + sep) + 3.5
        label = f"{i}"
        if r == 1:
            label = f"card_{c}"
        axes[1].text(x_center, y_center, label, ha="center", va="center",
                     color="cyan", fontsize=8, fontweight="bold")

    fig.suptitle("Card-game obs as 7x7 patches (patches 5-9 are the cards)")
    fig.tight_layout()
    out_path = "/tmp/patch_test.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved visualisation: {out_path}")


if __name__ == "__main__":
    main()
