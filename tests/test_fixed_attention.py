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

    # Image and feature map dims
    img_h = GRID_ROWS * TILE_PIXELS  # 21
    img_w = GRID_COLS * TILE_PIXELS  # 35
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w, stride=2, kernel_size=3, padding="SAME", num_blocks=4,
    )
    print(f"Image: {img_h}x{img_w}, Feature map: {feat_h}x{feat_w}")

    # Render a fixed card layout (no shuffle)
    perm = jnp.arange(NUM_CARDS)
    img = render_card_game(perm)
    img_np = np.array(img)

    scale = 32
    base_pil = Image.fromarray(img_np).resize(
        (img_w * scale, img_h * scale), Image.NEAREST
    )

    for pos in range(NUM_CARDS):
        # Create fixed attention map (same as in ja_ippo.py)
        attn = np.zeros((feat_h, feat_w), dtype=np.float32)
        card_row = feat_h // 3
        card_col = round(pos * feat_w / 5 + feat_w / 10)
        card_col = min(card_col, feat_w - 1)
        attn[card_row, card_col] = 1.0

        print(f"Position {pos}: feature map ({card_row}, {card_col})")
        print(f"  Attention map:\n{attn}")

        # Upsample attention to image size
        attn_resized = np.array(
            Image.fromarray(attn, mode='F').resize(
                (img_w * scale, img_h * scale), resample=Image.NEAREST
            )
        )

        # Overlay: red channel where attention is nonzero
        frame = np.array(base_pil).copy()
        mask = attn_resized > 0.5
        frame[mask] = [255, 0, 0]

        # Also draw a semi-transparent overlay
        alpha = 0.5
        overlay = frame.copy()
        import matplotlib.cm as cm
        heatmap = cm.Reds(attn_resized)[:, :, :3] * 255
        blended = (1 - alpha * attn_resized[:, :, None]) * frame + alpha * attn_resized[:, :, None] * heatmap
        result = np.clip(blended, 0, 255).astype(np.uint8)

        path = out_dir / f"card_pos_{pos}_attn_row{card_row}_col{card_col}.png"
        Image.fromarray(result).save(path)
        print(f"  Saved {path}")

    print(f"\nAll maps saved to {out_dir}/")
