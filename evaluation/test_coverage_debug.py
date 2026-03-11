"""Quick test to visually verify the feature-map → tile coverage mapping.

Creates an env, resets, runs one forward pass with random params to get an
attention map, builds the coverage map, and saves a debug image.

Usage:
    uv run python evaluation/test_coverage_debug.py

Output: evaluation/coverage_debug_agent_0.png, evaluation/coverage_debug_agent_1.png
"""

import jax
import jax.numpy as jnp
import numpy as np

from agents.initialize_agents import initialize_ja_image_agent, _get_image_dims
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import (
    build_coverage_map,
    render_coverage_debug,
    render_episode_frames,
)


def main():
    # Cramped room, image observations
    env_kwargs = {"layout": "cramped_room", "max_steps": 400}
    env = make_env("overcooked-v1", env_kwargs)

    if isinstance(env, LogWrapper):
        inner_env = env._env
    else:
        inner_env = env

    # Reset env to get one state
    rng = jax.random.PRNGKey(0)
    obs, env_state = inner_env.reset(rng)

    # Init policy with random params
    config = {
        "CONV_FILTERS": 32, "CONV_NUM_BLOCKS": 4, "CONV_KERNEL_SIZE": 3,
        "CONV_STRIDE": 2, "CONV_PADDING": "SAME",
        "JA_NUM_HEADS": 4, "JA_HEAD_FEATURES": 16,
        "FC_HIDDEN_DIM": 64, "LSTM_HIDDEN_DIM": 64,
        "JA_SPATIAL_BASIS_DEPTH": 8,
    }
    rng, init_rng = jax.random.split(rng)
    policy, init_params = initialize_ja_image_agent(config, inner_env, init_rng)

    # Compute feature map dims
    img_h, img_w, _ = _get_image_dims(inner_env)
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w, stride=2, kernel_size=3, padding="SAME", num_blocks=4)
    print(f"Image: {img_h}x{img_w}, Feature map: {feat_h}x{feat_w}")

    # Run one forward pass per agent to get attention maps
    hstate = policy.init_hstate(1)
    done = jnp.zeros((1, 1), dtype=bool)

    for agent_id, agent_name in enumerate(("agent_0", "agent_1")):
        agent_obs = obs[agent_name].reshape(1, 1, -1)
        rng, act_rng = jax.random.split(rng)

        _, _, attn = policy.get_action_and_attention(
            params=init_params, obs=agent_obs, done=done,
            avail_actions=jnp.ones((1, inner_env.action_space(agent_name).n)),
            hstate=hstate, rng=act_rng, agent_id=agent_id,
        )
        attn_map = np.array(attn).squeeze()
        print(f"{agent_name} attention shape: {attn_map.shape}, "
              f"sum: {attn_map.sum():.4f}")

        # Build coverage map
        coverage = build_coverage_map(env_state, feat_h, feat_w)

        # Render game frame
        frames = render_episode_frames([env_state], inner_env.agent_view_size,
                                       pixels_per_tile=32)
        frame = frames[0]

        # Save debug image
        debug_img = render_coverage_debug(frame, coverage, attn_map=attn_map)
        from PIL import Image
        out_path = f"evaluation/coverage_debug_{agent_name}.png"
        Image.fromarray(debug_img).save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
