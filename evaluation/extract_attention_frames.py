"""Extract attention overlay frames at consecutive timesteps for visualization.

Produces a 2x2 grid: rows = timestep t and t+1, columns = agent 0 and agent 1.
Each cell shows the game frame with attention heatmap overlay.

Usage:
    uv run python -m evaluation.extract_attention_frames \
        --checkpoint <path_to_saved_train_run> \
        --seed 4 \
        --timestep 50 \
        --output attention_grid.png
"""
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from agents.initialize_agents import (
    initialize_ja_image_agent, initialize_ja_dual_image_agent,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import (
    run_episode_with_states, render_episode_frames, _overlay_attention,
)


def _get_obs_type(alg_config):
    return alg_config.get("OBS_TYPE", alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Seed index within checkpoint")
    parser.add_argument("--timestep", type=int, default=50, help="Timestep t (will also show t+1)")
    parser.add_argument("--episode-rng", type=int, default=42, help="RNG seed for the episode")
    parser.add_argument("--output", default="attention_grid.png")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # Load config
    run_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg_config = cfg["algorithm"]

    env_name = alg_config["ENV_NAME"]
    env = make_env(env_name, alg_config["ENV_KWARGS"])
    inner_env = env
    env_wrapped = LogWrapper(env)

    obs_type = _get_obs_type(alg_config)
    use_dual = alg_config.get("USE_DUAL_CRITIC", False)
    if obs_type in ("image", "fov"):
        init_fn = initialize_ja_dual_image_agent if use_dual else initialize_ja_image_agent
    else:
        from agents.initialize_agents import initialize_ja_agent
        init_fn = initialize_ja_agent

    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(alg_config, env_wrapped, rng)

    run_data = load_train_run(args.checkpoint)
    final_params = run_data["final_params"]
    params = jax.tree.map(lambda x: x[args.seed], final_params)

    max_steps = int(alg_config.get("ENV_KWARGS", {}).get("max_steps", 400))

    print(f"Running episode (seed={args.seed}, rng={args.episode_rng})...")
    ep_states, attn_data = run_episode_with_states(
        jax.random.PRNGKey(args.episode_rng), inner_env, params, policy,
        params, policy, max_steps,
        collect_attention=True, greedy=True,
    )
    print(f"Episode: {len(ep_states)} frames, {len(attn_data['agent_0'])} attention maps")

    t = args.timestep
    if t + 1 >= len(attn_data["agent_0"]):
        print(f"Timestep {t} too large, episode has {len(attn_data['agent_0'])} steps. Using last 2.")
        t = len(attn_data["agent_0"]) - 2

    # Render game frames
    if env_name in ("lbf", "lbf-image", "lbf-reward-shaping"):
        from marl.ja_ippo import _render_lbf_eval_frames
        frames = _render_lbf_eval_frames(inner_env, ep_states)
    else:
        frames = render_episode_frames(ep_states, inner_env.agent_view_size, pixels_per_tile=32)

    # Debug: print attention map stats
    for agent_name in ["agent_0", "agent_1"]:
        attn = np.array(attn_data[agent_name][t]).squeeze()
        print(f"  {agent_name} t={t}: shape={attn.shape}, min={attn.min():.4f}, max={attn.max():.4f}, sum={attn.sum():.4f}")

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, timestep in enumerate([t, t + 1]):
        frame = frames[timestep]
        for col, agent_name in enumerate(["agent_0", "agent_1"]):
            ax = axes[row, col]
            attn_map = np.array(attn_data[agent_name][timestep]).squeeze()

            # Overlay attention on frame
            cmap = "Blues" if agent_name == "agent_0" else "Reds"
            overlaid = _overlay_attention(frame, attn_map, cmap, alpha=0.7)

            ax.imshow(overlaid)
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                agent_label = "Agent 0 (Blue)" if agent_name == "agent_0" else "Agent 1 (Red)"
                ax.set_title(agent_label, fontsize=16)
            if col == 0:
                ax.set_ylabel(f"t", fontsize=16) if row == 0 else ax.set_ylabel(f"t + 1", fontsize=16)

    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
