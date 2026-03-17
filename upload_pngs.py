"""Generate training curve PNGs from saved checkpoints and upload to existing wandb runs.

Usage:
    uv run python upload_pngs.py <checkpoint_path> <wandb_run_id>
"""
import argparse
import os

import jax
import numpy as np
import wandb

from common.plot_utils import get_metric_names, get_stats, plot_seed_aggregate
from common.save_load_utils import load_train_run
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-id", required=True, help="Existing wandb run ID")
    parser.add_argument("--project", default="aht-benchmark")
    parser.add_argument("--entity", default="g-benintendi-university-of-brescia")
    args = parser.parse_args()

    # Load config and metrics
    run_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg_config = cfg["algorithm"]

    run_data = load_train_run(args.checkpoint)
    train_metrics = run_data["metrics"]

    env_name = alg_config["ENV_NAME"]
    rollout_length = int(alg_config["ROLLOUT_LENGTH"])
    num_envs = int(alg_config["NUM_ENVS"])

    metric_names = get_metric_names(env_name)
    train_stats = get_stats(train_metrics, metric_names)

    # Generate PNGs
    plot_seed_aggregate(
        train_stats,
        num_rollout_steps=rollout_length,
        num_envs=num_envs,
        savedir=run_dir,
        savename="train_curve",
    )

    # Upload to existing run
    wb_run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        resume="must",
    )
    for name in train_stats:
        png_path = os.path.join(run_dir, f"train_curve_{name}.png")
        if os.path.exists(png_path):
            wb_run.log({f"Plots/train_curve_{name}": wandb.Image(png_path)}, commit=False)
            print(f"Uploaded {png_path}")
    wb_run.log({}, commit=True)
    wb_run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
