"""Quick test to visually verify the feature-map → tile coverage mapping.

Creates an env for each layout, resets, builds the coverage map, and saves
a debug image with the grid labels overlaid on the game frame.

Usage:
    ./run_gpu.sh 0 evaluation.test_coverage_debug

Output: evaluation/coverage_debug_<layout>.png for each layout
"""

import jax
import numpy as np

from agents.ja_image_actor_critic import _compute_resnet_output_dims
from envs import make_env
from evaluation.vis_episodes import (
    build_coverage_map,
    render_coverage_debug,
    render_episode_frames,
)

LAYOUTS = ["cramped_room", "coord_ring", "forced_coord", "asymm_advantages"]


def test_layout(layout_name):
    print(f"\n{'='*60}")
    print(f"Layout: {layout_name}")
    print(f"{'='*60}")

    env_kwargs = {"layout": layout_name, "max_steps": 400, "obs_type": "image"}
    env = make_env("overcooked-v1", env_kwargs)

    img_h = env.grid_height * env.tile_size
    img_w = env.grid_width * env.tile_size
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w, stride=2, kernel_size=3, padding="SAME", num_blocks=4)
    print(f"Grid: {env.grid_height}x{env.grid_width}, "
          f"Image: {img_h}x{img_w}, Feature map: {feat_h}x{feat_w}")

    rng = jax.random.PRNGKey(0)
    _, env_state = env.reset(rng)

    coverage = build_coverage_map(env_state, feat_h, feat_w)

    frames = render_episode_frames([env_state], env.agent_view_size,
                                   pixels_per_tile=32)
    frame = frames[0]

    from PIL import Image
    debug_img = render_coverage_debug(frame, coverage, attn_map=None)
    out_path = f"evaluation/coverage_debug_{layout_name}.png"
    Image.fromarray(debug_img).save(out_path)
    print(f"Saved: {out_path} ({debug_img.shape[1]}x{debug_img.shape[0]})")


def main():
    for layout in LAYOUTS:
        test_layout(layout)


if __name__ == "__main__":
    main()
