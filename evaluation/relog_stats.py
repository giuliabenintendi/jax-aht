"""Re-log a saved checkpoint to a new wandb run with clean metric names.

Mirrors the original training run: training curves with mean±std,
CSV export, and eval videos with attention overlays.

Usage:
    ./run_gpu.sh 3 evaluation.relog_stats \
        --checkpoint results/.../saved_train_run \
        --env-name overcooked-v1
"""
import argparse
import csv
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import wandb
import yaml

from agents.initialize_agents import initialize_ja_image_agent, initialize_ja_agent
from common.plot_utils import get_metric_names, get_stats
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import (
    run_episode_with_states, make_attention_video,
    render_episode_frames, build_coverage_map, INDEX_TO_OBJECT,
)


# Must match log_metrics() in ja_ippo.py
SCALAR_KEYS = [
    ("ja_beta",              "JA/beta"),
    ("jsd_mean",             "JA/jsd"),
    ("raw_env_reward_mean",  "Reward/env_raw"),
    ("combined_reward_mean", "Reward/combined_raw"),
    ("loss_total",           "Loss/total"),
    ("loss_value",           "Loss/value"),
    ("loss_policy",          "Loss/policy"),
    ("entropy",              "Loss/entropy"),
    ("grad_norm",            "Loss/grad_norm"),
    ("value_mean",           "Value/mean"),
]


def _load_hydra_config(checkpoint_path: str) -> dict:
    """Load the Hydra config.yaml from the run directory."""
    run_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _infer_run_name(cfg: dict) -> str:
    """Rebuild the run name suffix from config (same logic as _build_run_string)."""
    alg = cfg.get("algorithm", {})
    parts = [str(alg.get("ALG", "unknown"))]
    total = alg.get("TOTAL_TIMESTEPS")
    if total is not None:
        total = float(total)
        if total >= 1e6:
            parts.append(f"{total / 1e6:.0f}M")
        else:
            parts.append(f"{total:.0f}")
    beta = alg.get("JA_BETA_MAX")
    if beta is not None:
        parts.append(f"BETA{beta}")
    return "_".join(parts)


def _get_obs_type(alg_config: dict) -> str:
    return alg_config.get("OBS_TYPE", alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))


def _relog_training_metrics(train_metrics, env_name, run_dir,
                            rollout_length, num_envs):
    """Log training metrics to wandb and export CSV."""
    metric_names = get_metric_names(env_name)
    train_stats = get_stats(train_metrics, metric_names)
    episode_stats_mean = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    scalar_mean = {}
    scalar_std = {}
    for key, _ in SCALAR_KEYS:
        if key in train_metrics:
            vals = np.array(train_metrics[key])
            scalar_mean[key] = np.mean(vals, axis=0)
            scalar_std[key] = np.std(vals, axis=0)

    num_updates = train_metrics["returned_episode"].shape[1]
    num_seeds = train_metrics["returned_episode"].shape[0]

    # Export CSV
    csv_header = ["update", "timestep"]
    for name in metric_names:
        csv_header.extend([f"{name}_mean", f"{name}_std"])
    if env_name == "overcooked-v1" and "base_return" in metric_names:
        csv_header.append("soups_delivered")
    for key, _ in SCALAR_KEYS:
        if key in scalar_mean:
            csv_header.extend([f"{key}_mean", f"{key}_std"])

    csv_path = os.path.join(run_dir, "train_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for step in range(num_updates):
            row = [step, (step + 1) * rollout_length * num_envs]
            for name in metric_names:
                stat_data = np.array(train_stats[name])
                seed_means = stat_data[:, step, 0]
                row.extend([float(seed_means.mean()), float(seed_means.std())])
            if env_name == "overcooked-v1" and "base_return" in metric_names:
                base_data = np.array(train_stats["base_return"])
                row.append(float(base_data[:, step, 0].mean()) / 20.0)
            for key, _ in SCALAR_KEYS:
                if key in scalar_mean:
                    row.extend([float(scalar_mean[key][step]), float(scalar_std[key][step])])
            writer.writerow(row)

    print(f"[relog] CSV: {csv_path} ({num_updates} updates, {num_seeds} seeds, {len(csv_header)} cols)")
    wandb.save(csv_path, base_path=run_dir)

    # Log to wandb
    for step in range(num_updates):
        log_dict = {}
        for stat_name, stat_data in episode_stats_mean.items():
            log_dict[f"Train/{stat_name}_mean"] = stat_data[step, 0]
            log_dict[f"Train/{stat_name}_std"] = stat_data[step, 1]
        if "base_return" in episode_stats_mean and env_name == "overcooked-v1":
            log_dict["Train/soups_delivered"] = episode_stats_mean["base_return"][step, 0] / 20.0
        for key, wandb_name in SCALAR_KEYS:
            if key in scalar_mean:
                log_dict[f"{wandb_name}/mean"] = float(scalar_mean[key][step])
                log_dict[f"{wandb_name}/std"] = float(scalar_std[key][step])
        wandb.log(log_dict, step=step)

    print(f"[relog] Training metrics logged: {num_updates} steps, {num_seeds} seeds")


def _relog_eval_videos(alg_config, final_params, run_dir):
    """Re-run eval episodes and log videos + attention to wandb."""
    from moviepy import ImageSequenceClip
    from agents.ja_image_actor_critic import _compute_resnet_output_dims
    from agents.initialize_agents import _get_image_dims

    env_name = alg_config["ENV_NAME"]
    env = make_env(env_name, alg_config["ENV_KWARGS"])
    env = LogWrapper(env)

    obs_type = _get_obs_type(alg_config)
    init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent

    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(alg_config, env, rng)

    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    inner_env = env._env
    max_steps = int(alg_config.get("ENV_KWARGS", {}).get("max_steps", 400))

    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], final_params)

        ep_states, attn_data = run_episode_with_states(
            jax.random.PRNGKey(42 + seed_idx), inner_env, params, policy,
            params, policy, max_steps, collect_attention=True,
        )
        print(f"[relog] Seed {seed_idx}: {len(ep_states)} frames collected")

        video_dir = os.path.join(run_dir, "videos", f"seed_{seed_idx}")
        os.makedirs(video_dir, exist_ok=True)

        # Render frames
        if env_name in ("lbf", "lbf-image", "lbf-reward-shaping"):
            from marl.ja_ippo import _render_lbf_eval_frames
            frames = _render_lbf_eval_frames(inner_env, ep_states)
        else:
            frames = render_episode_frames(ep_states, inner_env.agent_view_size, pixels_per_tile=32)

        # Game video
        video_path = os.path.join(video_dir, "eval_final.mp4")
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(video_path, fps=10, codec='libx264', audio=False,
                             bitrate='8000k', preset='slow')
        tag = f"Eval/seed_{seed_idx}"
        wandb.log({f"{tag}/episode_video": wandb.Video(video_path, format="mp4")}, commit=False)

        # Attention overlay videos
        attn_video_base = os.path.join(video_dir, "eval_attention.mp4")
        make_attention_video(frames, attn_data, filename=attn_video_base, fps=10)
        for suffix in ("agent0", "agent1", "combined"):
            vpath = os.path.join(video_dir, f"eval_attention_{suffix}.mp4")
            wandb.log({f"{tag}/attention_{suffix}": wandb.Video(vpath, format="mp4")}, commit=False)

        # Coverage JSON (overcooked only)
        if env_name in ("overcooked-v1",):
            img_h, img_w, _ = _get_image_dims(env)
            feat_h, feat_w = _compute_resnet_output_dims(
                img_h, img_w,
                stride=alg_config.get("CONV_STRIDE", 2),
                kernel_size=alg_config.get("CONV_KERNEL_SIZE", 3),
                padding=alg_config.get("CONV_PADDING", "SAME"),
                num_blocks=alg_config.get("CONV_NUM_BLOCKS", 4),
            )
            attn_threshold = 0.05
            n_steps = min(len(attn_data["agent_0"]), len(ep_states) - 1)
            coverage_data = {}
            for agent_name in ("agent_0", "agent_1"):
                maps = attn_data[agent_name]
                agent_steps = []
                for t in range(n_steps):
                    attn = np.array(maps[t]).squeeze()
                    coverage = build_coverage_map(ep_states[t], feat_h, feat_w)
                    spots = []
                    for r in range(feat_h):
                        for c in range(feat_w):
                            val = float(attn[r, c])
                            if val >= attn_threshold:
                                cov = coverage[r, c]
                                cov_dict = {
                                    INDEX_TO_OBJECT[i]: round(float(cov[i]), 3)
                                    for i in range(len(INDEX_TO_OBJECT))
                                    if cov[i] > 0.01
                                }
                                spots.append({"attn": round(val, 4), "coverage": cov_dict})
                    spots.sort(key=lambda s: -s["attn"])
                    agent_steps.append({"t": t, "spots": spots})
                coverage_data[agent_name] = agent_steps

            json_path = os.path.join(video_dir, "attention_coverage.json")
            with open(json_path, "w") as f:
                json.dump(coverage_data, f, indent=2)
            print(f"[relog] Seed {seed_idx} coverage saved to {json_path}")

    print(f"[relog] Eval videos logged for {num_seeds} seeds")


def relog(checkpoint_path: str, env_name: str | None = None, run_name: str | None = None,
          project: str = "aht-benchmark",
          entity: str = "g-benintendi-university-of-brescia",
          rollout_length: int = 400, num_envs: int = 64,
          skip_videos: bool = False):
    """Full relog: training metrics + CSV + eval videos."""
    cfg = _load_hydra_config(checkpoint_path)
    alg_config = cfg["algorithm"]
    if env_name is None:
        env_name = alg_config["ENV_NAME"]
    run_dir = os.path.dirname(checkpoint_path)

    run_data = load_train_run(checkpoint_path)

    # Init wandb with same naming convention as training
    run_suffix = run_name or _infer_run_name(cfg)
    wb_run = wandb.init(
        project=project,
        entity=entity,
        config=alg_config,
        tags=["relog", f"seeds={run_data['metrics']['returned_episode'].shape[0]}"],
    )
    wb_run.name = str(wb_run.name) + "___" + run_suffix

    # Training metrics + CSV
    _relog_training_metrics(run_data["metrics"], env_name, run_dir,
                           rollout_length, num_envs)

    # Eval videos
    if not skip_videos:
        _relog_eval_videos(alg_config, run_data["final_params"], run_dir)

    wb_run.finish()
    print(f"[relog] Done -> {wb_run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-log a saved run to wandb with clean metrics + videos")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved_train_run directory")
    parser.add_argument("--env-name", default=None,
                        help="Environment name (default: inferred from Hydra config)")
    parser.add_argument("--run-name", default=None,
                        help="Name suffix for wandb run (default: inferred from Hydra config)")
    parser.add_argument("--project", default="aht-benchmark")
    parser.add_argument("--entity", default="g-benintendi-university-of-brescia")
    parser.add_argument("--rollout-length", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--skip-videos", action="store_true",
                        help="Skip eval video generation (metrics + CSV only)")
    args = parser.parse_args()

    relog(args.checkpoint, args.env_name, args.run_name,
          project=args.project, entity=args.entity,
          rollout_length=args.rollout_length, num_envs=args.num_envs,
          skip_videos=args.skip_videos)
