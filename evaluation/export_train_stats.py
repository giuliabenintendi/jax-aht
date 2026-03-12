"""Export training stats (mean ± std across seeds) to CSV.

Loads a saved_train_run checkpoint and writes per-update stats
to a CSV file for use in papers/analysis.

Usage:
    uv run python -m evaluation.export_train_stats \
        --checkpoint results/.../saved_train_run \
        --env-name overcooked-v1

    # Process all beta_sweep runs for a layout:
    uv run python -m evaluation.export_train_stats \
        --checkpoint results/overcooked-v1/cramped_room/ja_ippo/beta_sweep/2026-03-11_16-28-18/saved_train_run \
        --env-name overcooked-v1
"""
import argparse
import csv
import os

import numpy as np

from common.plot_utils import get_metric_names, get_stats
from common.save_load_utils import load_train_run


def export_stats(checkpoint_path: str, env_name: str, output_path: str | None = None,
                 rollout_length: int = 400, num_envs: int = 64):
    run_data = load_train_run(checkpoint_path)
    train_metrics = run_data["metrics"]

    metric_names = get_metric_names(env_name)
    train_stats = get_stats(train_metrics, metric_names)

    num_seeds = train_metrics["returned_episode"].shape[0]
    num_updates = train_metrics["returned_episode"].shape[1]

    # Scalar metrics: mean and std across seeds
    scalar_keys = [
        "ja_beta", "jsd_mean",
        "raw_env_reward_mean", "combined_reward_mean",
        "loss_total", "loss_value", "loss_policy", "entropy", "grad_norm",
        "value_mean",
    ]

    scalar_mean = {}
    scalar_std = {}
    for key in scalar_keys:
        if key in train_metrics:
            vals = np.array(train_metrics[key])
            scalar_mean[key] = np.mean(vals, axis=0)
            scalar_std[key] = np.std(vals, axis=0)

    # Build CSV
    if output_path is None:
        run_dir = os.path.dirname(checkpoint_path)
        output_path = os.path.join(run_dir, "train_stats.csv")

    # Header: timestep, then mean/std for each episode metric, then mean/std for each scalar
    header = ["update", "timestep"]
    for name in metric_names:
        header.extend([f"{name}_mean", f"{name}_std"])
    if env_name == "overcooked-v1" and "base_return" in metric_names:
        header.append("soups_delivered")
    for key in scalar_keys:
        if key in scalar_mean:
            header.extend([f"{key}_mean", f"{key}_std"])

    rows = []
    for step in range(num_updates):
        timestep = (step + 1) * rollout_length * num_envs
        row = [step, timestep]

        for name in metric_names:
            stat_data = np.array(train_stats[name])
            seed_means = stat_data[:, step, 0]
            row.append(float(seed_means.mean()))
            row.append(float(seed_means.std()))

        if env_name == "overcooked-v1" and "base_return" in metric_names:
            base_data = np.array(train_stats["base_return"])
            soups = float(base_data[:, step, 0].mean()) / 20.0
            row.append(soups)

        for key in scalar_keys:
            if key in scalar_mean:
                row.append(float(scalar_mean[key][step]))
                row.append(float(scalar_std[key][step]))

        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[export] {output_path}: {num_updates} updates, {num_seeds} seeds, {len(header)} columns")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export training stats to CSV")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved_train_run directory")
    parser.add_argument("--env-name", required=True,
                        help="Environment name (e.g. overcooked-v1)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <run_dir>/train_stats.csv)")
    parser.add_argument("--rollout-length", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=64)
    args = parser.parse_args()

    export_stats(args.checkpoint, args.env_name, args.output,
                 args.rollout_length, args.num_envs)
