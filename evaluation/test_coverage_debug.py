"""Quick test to visually verify the feature-map → tile coverage mapping.

Creates an env, resets, builds the coverage map, and saves a debug image
with the grid labels overlaid on the game frame.

Usage:
    ./run_gpu.sh 0 evaluation.test_coverage_debug

Output: evaluation/coverage_debug.png
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


def main():
    # Cramped room, image observations
    env_kwargs = {"layout": "cramped_room", "max_steps": 400, "obs_type": "image"}
    env = make_env("overcooked-v1", env_kwargs)

    img_h = env.grid_height * env.tile_size
    img_w = env.grid_width * env.tile_size
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w, stride=2, kernel_size=3, padding="SAME", num_blocks=4)
    print(f"Image: {img_h}x{img_w}, Feature map: {feat_h}x{feat_w}")

    # Reset env to get one state
    rng = jax.random.PRNGKey(0)
    obs, env_state = env.reset(rng)

    # Build coverage map
    coverage = build_coverage_map(env_state, feat_h, feat_w)

    # Print coverage for each cell
    from evaluation.vis_episodes import INDEX_TO_OBJECT
    for r in range(feat_h):
        for c in range(feat_w):
            cov = coverage[r, c]
            nonzero = {INDEX_TO_OBJECT[i]: f"{cov[i]:.0%}"
                       for i in range(len(cov)) if cov[i] > 0.01}
            print(f"  cell ({r},{c}): {nonzero}")

    # Render game frame
    frames = render_episode_frames([env_state], env.agent_view_size,
                                   pixels_per_tile=32)
    frame = frames[0]
    print(f"Frame shape: {frame.shape}")

    # Save debug image (no attention, all cells labeled)
    from PIL import Image
    debug_img = render_coverage_debug(frame, coverage, attn_map=None)
    out_path = "evaluation/coverage_debug.png"
    Image.fromarray(debug_img).save(out_path)
    print(f"Saved: {out_path} ({debug_img.shape[1]}x{debug_img.shape[0]})")


if __name__ == "__main__":
    main()
