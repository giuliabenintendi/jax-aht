"""Re-log training stats (mean + std) to wandb from a saved checkpoint.

Loads the saved_train_run, computes per-seed stats, and logs mean/std
to a new wandb run with the same naming as log_metrics in ja_ippo.py.
Also exports train_stats.csv and uploads it as a wandb artifact.

Usage:
    uv run python -m evaluation.relog_stats \
        --checkpoint results/.../saved_train_run \
        --env-name overcooked-v1 \
        --run-name "cramped_room_beta0_relog"
"""
import argparse
import csv
import os

import numpy as np
import wandb

from common.plot_utils import get_metric_names, get_stats
from common.save_load_utils import load_train_run


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


def relog_stats(checkpoint_path: str, env_name: str, run_name: str,
                project: str = "aht-benchmark",
                entity: str = "g-benintendi-university-of-brescia",
                rollout_length: int = 400, num_envs: int = 64):
    run_data = load_train_run(checkpoint_path)
    train_metrics = run_data["metrics"]

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
    run_dir = os.path.dirname(checkpoint_path)
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

    # Log to wandb
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=["relog", f"seeds={num_seeds}"],
    )

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

    wandb.save(csv_path, base_path=run_dir)

    run.finish()
    print(f"[relog] Done: {num_updates} steps, {num_seeds} seeds -> {run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-log training stats with std to wandb")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved_train_run directory")
    parser.add_argument("--env-name", required=True,
                        help="Environment name (e.g. overcooked-v1)")
    parser.add_argument("--run-name", required=True,
                        help="Name for the wandb run")
    parser.add_argument("--project", default="aht-benchmark")
    parser.add_argument("--entity", default="g-benintendi-university-of-brescia")
    parser.add_argument("--rollout-length", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=64)
    args = parser.parse_args()

    relog_stats(args.checkpoint, args.env_name, args.run_name,
                project=args.project, entity=args.entity,
                rollout_length=args.rollout_length, num_envs=args.num_envs)
