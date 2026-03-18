"""Add eval videos to an existing wandb run from a saved checkpoint.

Usage:
    uv run python -m evaluation.add_eval_videos \
        --checkpoint <path_to_saved_train_run> \
        --run-id <wandb_run_id>
"""
import argparse
import os

import jax
import wandb
from omegaconf import OmegaConf

from agents.initialize_agents import (
    initialize_ja_image_agent, initialize_ja_dual_image_agent,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import (
    run_episode_with_states, make_attention_video, render_episode_frames,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-id", required=True, help="Existing wandb run ID to resume")
    parser.add_argument("--project", default="aht-benchmark")
    parser.add_argument("--entity", default="g-benintendi-university-of-brescia")
    args = parser.parse_args()

    # Load config
    run_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg_config = cfg["algorithm"]

    env_name = alg_config["ENV_NAME"]
    env = make_env(env_name, alg_config["ENV_KWARGS"])
    env = LogWrapper(env)

    # Detect dual critic and init policy
    obs_type = alg_config.get("OBS_TYPE", alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))
    use_dual = alg_config.get("USE_DUAL_CRITIC", False)
    if obs_type in ("image", "fov"):
        init_fn = initialize_ja_dual_image_agent if use_dual else initialize_ja_image_agent
    else:
        from agents.initialize_agents import initialize_ja_agent
        init_fn = initialize_ja_agent

    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(alg_config, env, rng)

    # Load params
    run_data = load_train_run(args.checkpoint)
    final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]

    inner_env = env._env
    max_steps = int(alg_config.get("ENV_KWARGS", {}).get("max_steps", 400))

    # Resume wandb run
    wb_run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        resume="must",
    )

    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], final_params)

        ep_states, attn_data = run_episode_with_states(
            jax.random.PRNGKey(42 + seed_idx), inner_env, params, policy,
            params, policy, max_steps,
            collect_attention=True,
        )
        print(f"Seed {seed_idx}: {len(ep_states)} frames collected")

        video_dir = os.path.join(run_dir, "videos", f"seed_{seed_idx}")
        os.makedirs(video_dir, exist_ok=True)

        # Render frames
        if env_name in ("lbf", "lbf-image", "lbf-reward-shaping"):
            from marl.ja_ippo import _render_lbf_eval_frames
            frames = _render_lbf_eval_frames(inner_env, ep_states)
        else:
            frames = render_episode_frames(ep_states, inner_env.agent_view_size, pixels_per_tile=32)

        # Game video
        from moviepy import ImageSequenceClip
        video_path = os.path.join(video_dir, "eval_final.mp4")
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(video_path, fps=10, codec='libx264', audio=False,
                             bitrate='8000k', preset='slow')
        tag = f"Eval/seed_{seed_idx}"
        wb_run.log({f"{tag}/episode_video": wandb.Video(video_path, format="mp4")}, commit=False)

        # Attention overlay videos
        attn_video_base = os.path.join(video_dir, "eval_attention.mp4")
        make_attention_video(frames, attn_data, filename=attn_video_base, fps=10)
        for suffix in ("agent0", "agent1", "combined"):
            vpath = os.path.join(video_dir, f"eval_attention_{suffix}.mp4")
            wb_run.log({f"{tag}/attention_{suffix}": wandb.Video(vpath, format="mp4")}, commit=False)

    wb_run.log({}, commit=True)
    wb_run.finish()
    print(f"Eval videos added to {wb_run.url}")


if __name__ == "__main__":
    main()
