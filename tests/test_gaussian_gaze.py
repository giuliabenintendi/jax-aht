"""Geometry and visualization tests for Gaussian gaze maps.

Run examples:
  uv run pytest -q tests/test_gaussian_gaze.py::test_gaussian_attention_properties
  uv run pytest -s tests/test_gaussian_gaze.py::test_save_card_game_gaussian_gaze_overlays
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from agents.gaze_image_actor_critic import gaussian_attention_from_params
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from envs import make_env


def _center_of_mass(attn):
    h, w = attn.shape
    ys = jnp.linspace(-1.0, 1.0, h)
    xs = jnp.linspace(-1.0, 1.0, w)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
    cx = jnp.sum(attn * grid_x)
    cy = jnp.sum(attn * grid_y)
    return cx, cy


def _overlay_heat(obs_flat, attn, img_h, img_w, out_path, alpha=0.55):
    base = np.array(obs_flat).reshape(img_h, img_w, 3)
    up = np.array(jax.image.resize(attn, (img_h, img_w), method="nearest"))
    up = up / max(float(up.max()), 1e-8)

    heat = np.zeros((img_h, img_w, 3), dtype=np.float32)
    heat[..., 0] = up
    heat[..., 1] = 0.2 * up

    overlay = np.clip((1.0 - alpha) * base + alpha * heat, 0.0, 1.0)
    img = (overlay * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


def test_gaussian_attention_properties():
    feat_h, feat_w = 9, 13
    attn = gaussian_attention_from_params(
        feat_h=feat_h,
        feat_w=feat_w,
        mu_x=jnp.array(0.0),
        mu_y=jnp.array(0.0),
        sigma_x=jnp.array(0.45),
        sigma_y=jnp.array(0.45),
        rho=jnp.array(0.0),
    )

    assert attn.shape == (feat_h, feat_w)
    assert jnp.allclose(attn.sum(), 1.0, atol=1e-6)

    cx, cy = _center_of_mass(attn)
    assert abs(float(cx)) < 0.08
    assert abs(float(cy)) < 0.08

    attn_left = gaussian_attention_from_params(
        feat_h, feat_w,
        mu_x=jnp.array(-0.75),
        mu_y=jnp.array(0.0),
        sigma_x=jnp.array(0.25),
        sigma_y=jnp.array(0.25),
        rho=jnp.array(0.0),
    )
    attn_right = gaussian_attention_from_params(
        feat_h, feat_w,
        mu_x=jnp.array(0.75),
        mu_y=jnp.array(0.0),
        sigma_x=jnp.array(0.25),
        sigma_y=jnp.array(0.25),
        rho=jnp.array(0.0),
    )
    cx_left, _ = _center_of_mass(attn_left)
    cx_right, _ = _center_of_mass(attn_right)
    assert float(cx_left) < -0.45
    assert float(cx_right) > 0.45

    attn_narrow = gaussian_attention_from_params(
        feat_h, feat_w,
        mu_x=jnp.array(0.0),
        mu_y=jnp.array(0.0),
        sigma_x=jnp.array(0.15),
        sigma_y=jnp.array(0.15),
        rho=jnp.array(0.0),
    )
    attn_wide = gaussian_attention_from_params(
        feat_h, feat_w,
        mu_x=jnp.array(0.0),
        mu_y=jnp.array(0.0),
        sigma_x=jnp.array(0.70),
        sigma_y=jnp.array(0.70),
        rho=jnp.array(0.0),
    )
    entropy_narrow = -jnp.sum(attn_narrow * jnp.log(attn_narrow + 1e-8))
    entropy_wide = -jnp.sum(attn_wide * jnp.log(attn_wide + 1e-8))
    assert float(entropy_wide) > float(entropy_narrow)


def test_save_card_game_gaussian_gaze_overlays():
    out_dir = Path("tests/gaussian_gaze_vis")
    out_dir.mkdir(exist_ok=True)

    env = make_env("card-game", {"max_steps": 4})
    obs, _ = env.reset(jax.random.PRNGKey(0))
    obs_0 = obs["agent_0"]

    img_h = env.grid_height * env.tile_size
    img_w = env.grid_width * env.tile_size
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w,
        stride=2,
        kernel_size=3,
        padding="SAME",
        num_blocks=4,
    )

    configs = [
        ("center_medium", 0.0, 0.0, 0.45, 0.45, 0.0),
        ("left_narrow", -0.7, 0.0, 0.20, 0.20, 0.0),
        ("right_wide", 0.7, 0.0, 0.65, 0.35, 0.0),
        ("diag_tilt", 0.0, 0.0, 0.35, 0.20, 0.65),
    ]

    for name, mu_x, mu_y, sx, sy, rho in configs:
        attn = gaussian_attention_from_params(
            feat_h, feat_w,
            mu_x=jnp.array(mu_x),
            mu_y=jnp.array(mu_y),
            sigma_x=jnp.array(sx),
            sigma_y=jnp.array(sy),
            rho=jnp.array(rho),
        )
        out_path = out_dir / f"{name}.png"
        _overlay_heat(obs_0, attn, img_h, img_w, out_path)
        cx, cy = _center_of_mass(attn)
        print(
            f"{name}: saved={out_path} sum={float(attn.sum()):.4f} "
            f"com=({float(cx):.3f},{float(cy):.3f}) max={float(attn.max()):.4f}"
        )

    assert (out_dir / "center_medium.png").exists()
