"""Print cross-seed final stats from a saved checkpoint.

Usage:
    uv run python -m evaluation.print_final_stats --checkpoint <path>
"""
import argparse

import jax
import numpy as np

from common.plot_utils import get_metric_names, get_stats
from common.save_load_utils import load_train_run
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    import os
    run_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg_config = cfg["algorithm"]

    run_data = load_train_run(args.checkpoint)
    train_metrics = run_data["metrics"]

    env_name = alg_config["ENV_NAME"]
    metric_names = get_metric_names(env_name)
    train_stats = get_stats(train_metrics, metric_names)

    num_seeds = train_metrics["returned_episode"].shape[0]
    print(f"Layout: {cfg.get('TASK_NAME', '?')}  Beta: {alg_config.get('JA_BETA_MAX', '?')}  Seeds: {num_seeds}")
    print()

    for name in metric_names:
        v = np.array(train_stats[name])  # (num_seeds, num_updates, 2)
        seed_final_means = v[:, -1, 0]   # per-seed final mean
        cross_mean = seed_final_means.mean()
        cross_std = seed_final_means.std()
        cross_sem = cross_std / np.sqrt(num_seeds)
        print(f"{name}:")
        print(f"  per-seed finals: {[f'{x:.1f}' for x in seed_final_means]}")
        print(f"  mean={cross_mean:.1f}  std={cross_std:.1f}  sem={cross_sem:.1f}")
        if name == "base_return" and env_name == "overcooked-v1":
            print(f"  soups={cross_mean / 20:.2f}")
    print()

    # Scalar metrics
    scalar_keys = ["jsd_mean", "loss_total", "entropy"]
    for key in scalar_keys:
        if key in train_metrics:
            vals = np.array(train_metrics[key])  # (num_seeds, num_updates)
            seed_finals = vals[:, -1]
            print(f"{key}: mean={seed_finals.mean():.4f}  std={seed_finals.std():.4f}")


if __name__ == "__main__":
    main()
