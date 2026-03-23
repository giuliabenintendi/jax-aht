"""Visualize the fixed attention map for each card position.

Renders the card game scene with the fixed attention overlaid
to verify the attention points at the correct card.

Run: ./run_gpu.sh 0 pytest -s tests/test_fixed_attention.py
"""
import numpy as np
import jax.numpy as jnp
from PIL import Image
from pathlib import Path

from envs.card_game.rendering import (
    render_card_game, TILE_PIXELS, GRID_ROWS, GRID_COLS, NUM_CARDS,
)
from agents.ja_image_actor_critic import _compute_resnet_output_dims


def test_fixed_attention_maps():
    """Save overlay images for each card position's fixed attention."""
    out_dir = Path("tests/fixed_attention_maps")
    out_dir.mkdir(exist_ok=True)

    img_h = GRID_ROWS * TILE_PIXELS  # 21
    img_w = GRID_COLS * TILE_PIXELS  # 35
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w, stride=2, kernel_size=3, padding="SAME", num_blocks=4,
    )
    print(f"Image: {img_h}x{img_w}, Feature map: {feat_h}x{feat_w}")

    perm = jnp.arange(NUM_CARDS)
    img = render_card_game(perm)
    img_np = np.array(img)

    scale = 32

    for pos in range(NUM_CARDS):
        # Same mapping as in ja_ippo.py
        pixel_col = pos * TILE_PIXELS + TILE_PIXELS // 2
        pixel_row = 1 * TILE_PIXELS + TILE_PIXELS // 2
        fc = min(round(pixel_col / (img_w / feat_w)), feat_w - 1)
        fr = min(round(pixel_row / (img_h / feat_h)), feat_h - 1)

        # Build attention with small blob
        attn = np.zeros((feat_h, feat_w), dtype=np.float32)
        attn[fr, fc] = 1.0
        if fr + 1 < feat_h:
            attn[fr + 1, fc] = 0.5
        attn /= attn.sum()

        print(f"Position {pos}: pixel ({pixel_row},{pixel_col}) -> feature ({fr},{fc})")

        # Upsample to image size
        attn_resized = np.array(
            Image.fromarray(attn, mode='F').resize(
                (img_w * scale, img_h * scale), resample=Image.NEAREST
            )
        )

        # Overlay
        base = np.array(Image.fromarray(img_np).resize(
            (img_w * scale, img_h * scale), Image.NEAREST
        ))
        import matplotlib.cm as cm
        a_norm = attn_resized / (attn_resized.max() + 1e-8)
        heatmap = (cm.Reds(a_norm)[:, :, :3] * 255).astype(np.uint8)
        alpha = 0.6
        blended = ((1 - alpha * a_norm[:, :, None]) * base + alpha * a_norm[:, :, None] * heatmap)
        result = np.clip(blended, 0, 255).astype(np.uint8)

        path = out_dir / f"card_pos_{pos}.png"
        Image.fromarray(result).save(path)
        print(f"  Saved {path}")
