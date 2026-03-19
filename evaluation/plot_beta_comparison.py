"""Plot beta comparison curves from wandb training runs.

Downloads training curves for JA-IPPO experiments across multiple layouts
and beta values, then produces plots matching default matplotlib style.

Usage:
    uv run python -m evaluation.plot_beta_comparison --output-dir plots/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "g-benintendi-university-of-brescia"
PROJECT = "aht-benchmark"

ROLLOUT_LENGTH = 400
NUM_ENVS = 64

LAYOUTS = {
    "Cramped Room": {
        0.0: "kjyyivlw",
        0.1: "62szp8y9",
        0.25: "cz1cw3xj",
        0.5: "g41q7gsw",
        1.0: "6ccfw6kc",
    },
    "Coord Ring": {
        0.0: "6ra3cuak",
        0.1: "2x8eo1up",
        0.25: "wq1yl77a",
        0.5: "t9knl6av",
        1.0: "01zbhuzy",
    },
    "Forced Coord": {
        0.0: "sjqd2q3k",
        0.1: "p1ezbvee",
        0.25: "c342bx87",
        0.5: "h144t2m7",
        1.0: "xl0jjs8a",
    },
}

# Default matplotlib tab colors
BETA_COLORS = {
    0.0: "C0",   # tab:blue
    0.1: "C1",   # tab:orange
    0.25: "C2",  # tab:green
    0.5: "C3",   # tab:red
    1.0: "C4",   # tab:purple
}

METRICS = {
    "base_return": {
        "seed_metric": "base_return",
        "ylabel": "Mean Episode Return",
    },
}


def fetch_run_data(api: wandb.Api, run_id: str, metric_name: str, num_seeds: int = 5):
    """Fetch per-seed training curves and compute cross-seed mean/std."""
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    seed_keys = [f"Seeds/{metric_name}/seed_{i}" for i in range(num_seeds)]
    rows = list(run.scan_history(keys=seed_keys))

    # Build (num_steps, num_seeds) array
    all_seeds = []
    for row in rows:
        vals = [row.get(k) for k in seed_keys]
        if any(v is None for v in vals):
            continue
        all_seeds.append(vals)

    all_seeds = np.array(all_seeds)  # (num_steps, num_seeds)
    steps = np.arange(len(all_seeds))
    timesteps = (steps + 1) * ROLLOUT_LENGTH * NUM_ENVS
    means = all_seeds.mean(axis=1)
    sem = all_seeds.std(axis=1) / np.sqrt(all_seeds.shape[1])
    return timesteps, means, sem


def plot_single_layout(
    ax,
    layout_name: str,
    run_ids: dict,
    metric_key: str,
    api: wandb.Api,
    cache: dict,
):
    """Plot all beta curves for one layout on the given axes."""
    metric_info = METRICS[metric_key]
    seed_metric = metric_info["seed_metric"]

    for beta, run_id in sorted(run_ids.items()):
        cache_key = (run_id, seed_metric)
        if cache_key not in cache:
            print(f"  fetching {layout_name} beta={beta} ({run_id})...")
            cache[cache_key] = fetch_run_data(api, run_id, seed_metric)
        timesteps, mean_plot, std_plot = cache[cache_key]

        color = BETA_COLORS[beta]
        label = f"β = {beta}"
        ax.plot(timesteps, mean_plot, color=color, linewidth=2.0, label=label)
        ax.fill_between(
            timesteps,
            mean_plot - std_plot,
            mean_plot + std_plot,
            color=color,
            alpha=0.25,
        )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(metric_info["ylabel"])
    ax.set_title(layout_name)
    ax.legend(loc="best")


def main():
    parser = argparse.ArgumentParser(description="Plot beta comparison from wandb runs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    cache = {}

    # Individual plots per layout, per metric
    for metric_key in METRICS:
        for layout_name, run_ids in LAYOUTS.items():
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_single_layout(ax, layout_name, run_ids, metric_key, api, cache)
            fig.tight_layout()

            slug = layout_name.lower().replace(" ", "_")
            filename = f"beta_comparison_{slug}_{metric_key}.png"
            fig.savefig(output_dir / filename, dpi=args.dpi)
            plt.close(fig)
            print(f"saved {output_dir / filename}")

    # Combined 3-panel figure for base_return (soups delivered)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, (layout_name, run_ids) in zip(axes, LAYOUTS.items()):
        plot_single_layout(ax, layout_name, run_ids, "base_return", api, cache)

    fig.tight_layout(w_pad=3.0)
    combined_path = output_dir / "beta_comparison_all_layouts.png"
    fig.savefig(combined_path, dpi=args.dpi)
    plt.close(fig)
    print(f"saved {combined_path}")


if __name__ == "__main__":
    main()
