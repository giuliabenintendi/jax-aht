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
        "mean": "Train/base_return_mean",
        "std": "Train/base_return_std",
        "ylabel": "Mean Episode Return",
    },
}


def fetch_run_data(api: wandb.Api, run_id: str, metric_mean: str, metric_std: str):
    """Fetch training curves from a wandb run (all rows, no truncation)."""
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    rows = list(run.scan_history(keys=[metric_mean, metric_std]))

    steps = []
    means = []
    stds = []
    for i, row in enumerate(rows):
        val = row.get(metric_mean)
        if val is None:
            continue
        steps.append(i)
        means.append(val)
        stds.append(row.get(metric_std, 0.0) or 0.0)

    steps = np.array(steps)
    timesteps = (steps + 1) * ROLLOUT_LENGTH * NUM_ENVS
    return timesteps, np.array(means), np.array(stds)


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
    scale = metric_info.get("scale", 1.0)

    for beta, run_id in sorted(run_ids.items()):
        cache_key = (run_id, metric_info["mean"])
        if cache_key not in cache:
            print(f"  fetching {layout_name} beta={beta} ({run_id})...")
            cache[cache_key] = fetch_run_data(
                api, run_id, metric_info["mean"], metric_info["std"]
            )
        timesteps, mean_vals, std_vals = cache[cache_key]

        mean_plot = mean_vals * scale
        std_plot = std_vals * scale

        color = BETA_COLORS[beta]
        label = f"β = {beta}"
        ax.plot(timesteps, mean_plot, color=color, linewidth=1.5, label=label)
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
