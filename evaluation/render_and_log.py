"""Standalone script to render episodes from saved checkpoints and log videos to WandB.

Usage:
    .venv/bin/python evaluation/render_and_log.py <results_dir> [--wandb-project ja-aht] [--wandb-entity ENTITY]

Example:
    .venv/bin/python evaluation/render_and_log.py \
        results/overcooked-v1/cramped_room/fcp/po_cone_fcp/2026-02-14_14-08-45 \
        --wandb-project ja-aht \
        --wandb-entity g-benintendi-university-of-brescia
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import wandb

from envs import make_env
from common.save_load_utils import load_train_run
from agents.initialize_agents import initialize_mlp_agent
from evaluation.vis_episodes import save_video


def main():
    parser = argparse.ArgumentParser(description="Render episodes and log to WandB")
    parser.add_argument("results_dir", help="Path to the results directory (contains saved_train_run, ego_train_run)")
    parser.add_argument("--wandb-project", default="ja-aht")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-id", default=None, help="Resume an existing WandB run by ID")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--po-mode", default="cone")
    parser.add_argument("--fov-range", type=int, default=5)
    parser.add_argument("--fov-slope", type=float, default=0.7)
    args = parser.parse_args()

    # Build env kwargs
    env_kwargs = {
        "layout": args.layout,
        "random_reset": True,
        "random_obj_state": True,
        "max_steps": 400,
        "po_mode": args.po_mode,
        "fov_range": args.fov_range,
        "fov_slope": args.fov_slope,
        "use_occlusion": False,
        "soft_view": True,
    }

    env = make_env("overcooked-v1", env_kwargs)
    rng = jax.random.PRNGKey(42)

    # Load FCP partner checkpoint (first seed, first partner's final params)
    fcp_path = os.path.join(args.results_dir, "saved_train_run")
    print(f"Loading FCP partners from {fcp_path}...")
    fcp_ckpt = load_train_run(fcp_path)
    # fcp_ckpt["checkpoints"] shape: (num_seeds, pop_size, num_ckpts, ...)
    # Take first seed, first partner, last checkpoint
    partner_params = jax.tree.map(lambda x: x[0, 0, -1], fcp_ckpt["checkpoints"])

    # Load ego checkpoint
    ego_path = os.path.join(args.results_dir, "ego_train_run")
    print(f"Loading ego agent from {ego_path}...")
    ego_ckpt = load_train_run(ego_path)
    # ego_ckpt["final_params"] shape: (num_seeds, num_ego_seeds, ...)
    ego_params = jax.tree.map(lambda x: x[0, 0], ego_ckpt["final_params"])

    # Initialize policies (MLP for FCP partners, S5 for ego — but both use same interface for get_action)
    rng, init1_rng, init2_rng = jax.random.split(rng, 3)
    # FCP partners are MLP
    partner_policy, _ = initialize_mlp_agent({}, env, init1_rng)

    # Ego is S5 — need to use the right policy
    from ego_agent_training.utils import initialize_ego_agent
    ego_config = {
        "EGO_ACTOR_TYPE": "s5",
        "S5_D_MODEL": 16,
        "S5_SSM_SIZE": 16,
        "S5_ACTOR_CRITIC_HIDDEN_DIM": 64,
        "FC_N_LAYERS": 2,
    }
    ego_policy, _ = initialize_ego_agent(ego_config, env, init2_rng)

    # Render videos
    video_dir = os.path.join(args.results_dir, "videos")
    print(f"Rendering {args.num_episodes} episodes...")
    video_path = save_video(
        env, "overcooked-v1",
        agent_0_param=ego_params, agent_0_policy=ego_policy,
        agent_1_param=partner_params, agent_1_policy=partner_policy,
        max_episode_steps=400, num_eps=args.num_episodes,
        savevideo=True, save_dir=video_dir, save_name="ego_vs_partner_po")
    print(f"Video saved at {video_path}")

    # Log to WandB
    wandb_kwargs = {
        "project": args.wandb_project,
        "mode": "online",
    }
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_id:
        wandb_kwargs["id"] = args.wandb_run_id
        wandb_kwargs["resume"] = "allow"

    run = wandb.init(**wandb_kwargs)
    wandb.log({"Videos/ego_vs_partner_po": wandb.Video(video_path)})
    print(f"Video logged to WandB: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
