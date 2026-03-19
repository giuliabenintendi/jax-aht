"""Plot beta comparison curves for LBF experiments (3-food and 10-food).

Downloads per-seed training curves from wandb and plots mean ± SEM.
For beta=0 (1 seed), shows a single line with no band.

Usage:
    uv run python -m evaluation.plot_lbf_beta_comparison --output-dir plots/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "g-benintendi-university-of-brescia"
PROJECT = "aht-benchmark"

ROLLOUT_LENGTH = 128
NUM_ENVS = 64

LAYOUTS = {
    "LBF 3-food": {
        0.0: {"run_id": "mj2f2u9p", "num_seeds": 3},
        0.001: {"run_id": "rfzfw022", "num_seeds": 3},
        0.002: {"run_id": "ef0q48cs", "num_seeds": 3},
    },
    "LBF 10-food": {
        0.0: {"run_id": "9zyi9pzn", "num_seeds": 3},
        0.001: {"run_id": "z6ve9ebv", "num_seeds": 3},
        0.002: {"run_id": "jnmsu3ry", "num_seeds": 3},
    },
}

BETA_COLORS = {
    0.0: "C0",
    0.001: "C1",
    0.002: "C2",
}


def fetch_run_data(api, run_id, num_seeds):
    """Fetch per-seed training curves. Falls back to mean/std if per-seed data missing."""
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    # Try per-seed data first
    seed_keys = [f"Seeds/returned_episode_returns/seed_{i}" for i in range(num_seeds)]
    rows = list(run.scan_history(keys=seed_keys))
    all_seeds = []
    for row in rows:
        vals = [row.get(k) for k in seed_keys]
        if any(v is None for v in vals):
            continue
        all_seeds.append(vals)

    if len(all_seeds) > 0:
        all_seeds = np.array(all_seeds)
        steps = np.arange(len(all_seeds))
        timesteps = (steps + 1) * ROLLOUT_LENGTH * NUM_ENVS
        means = all_seeds.mean(axis=1)
        sems = all_seeds.std(axis=1) / np.sqrt(num_seeds)
        print(f"    using per-seed data ({num_seeds} seeds, {len(all_seeds)} steps)")
        return timesteps, means, sems

    # Fallback: use mean/std from wandb
    rows = list(run.scan_history(keys=[
        "Train/returned_episode_returns_mean",
        "Train/returned_episode_returns_std",
    ]))
    means = []
    stds = []
    for row in rows:
        val = row.get("Train/returned_episode_returns_mean")
        if val is not None:
            means.append(val)
            stds.append(row.get("Train/returned_episode_returns_std", 0) or 0)
    means = np.array(means)
    stds = np.array(stds)
    steps = np.arange(len(means))
    timesteps = (steps + 1) * ROLLOUT_LENGTH * NUM_ENVS
    sems = stds / np.sqrt(num_seeds)
    print(f"    fallback to mean/std ({len(means)} steps, SEM from {num_seeds} seeds)")
    return timesteps, means, sems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    layout_names = list(LAYOUTS.keys())

    # Individual plots per layout
    for layout_name, run_configs in LAYOUTS.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for beta in sorted(run_configs.keys()):
            cfg = run_configs[beta]
            print(f"  fetching {layout_name} beta={beta} ({cfg['run_id']})...")
            timesteps, means, sems = fetch_run_data(api, cfg["run_id"], cfg["num_seeds"])

            color = BETA_COLORS[beta]
            label = f"β = {beta}"
            ax.plot(timesteps, means, color=color, linewidth=2.0, label=label)
            if sems.any():
                ax.fill_between(timesteps, means - sems, means + sems,
                                color=color, alpha=0.25)

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Episode Return")
        ax.set_title(layout_name)
        ax.legend(loc="lower right")

        slug = layout_name.lower().replace(" ", "_").replace("-", "_")
        path = output_dir / f"beta_comparison_{slug}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {path}")

    # Combined 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (layout_name, run_configs) in zip(axes, LAYOUTS.items()):
        for beta in sorted(run_configs.keys()):
            cfg = run_configs[beta]
            timesteps, means, sems = fetch_run_data(api, cfg["run_id"], cfg["num_seeds"])

            color = BETA_COLORS[beta]
            label = f"β = {beta}"
            ax.plot(timesteps, means, color=color, linewidth=2.0, label=label)
            if sems.any():
                ax.fill_between(timesteps, means - sems, means + sems,
                                color=color, alpha=0.25)

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Episode Return")
        ax.set_title(layout_name)
        ax.legend(loc="lower right")

    fig.tight_layout()

    combined_path = output_dir / "beta_comparison_lbf_all.png"
    fig.savefig(combined_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {combined_path}")


if __name__ == "__main__":
    main()
