"""Plot beta comparison curves from wandb training runs.

Downloads training curves for JA-IPPO experiments across multiple layouts
and beta values, then produces publication-quality PDF plots.

Usage:
    uv run python -m evaluation.plot_beta_comparison --output-dir plots/
"""

import argparse
import os
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

BETA_COLORS = {
    0.0: "#2166ac",
    0.1: "#66bd63",
    0.25: "#fee08b",
    0.5: "#f46d43",
    1.0: "#d73027",
}

METRICS = {
    "episode_return": {
        "mean": "Train/returned_episode_returns_mean",
        "std": "Train/returned_episode_returns_std",
        "ylabel": "Mean Episode Return",
    },
    "base_return": {
        "mean": "Train/base_return_mean",
        "std": "Train/base_return_std",
        "ylabel": "Soups Delivered",
        "scale": 1.0 / 20.0,
    },
}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average."""
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


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

        mean_smooth = ema(mean_vals * scale)
        std_smooth = ema(std_vals * scale)

        color = BETA_COLORS[beta]
        label = rf"$\beta = {beta}$"
        ax.plot(timesteps, mean_smooth, color=color, linewidth=1.8, label=label)
        ax.fill_between(
            timesteps,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            color=color,
            alpha=0.18,
        )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(metric_info["ylabel"])
    ax.set_title(layout_name)
    ax.legend(
        loc="best",
        frameon=True,
        edgecolor="0.7",
        fancybox=False,
        framealpha=0.9,
    )
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))


def main():
    parser = argparse.ArgumentParser(description="Plot beta comparison from wandb runs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save PDF plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    api = wandb.Api()
    cache = {}

    # Individual plots per layout, per metric
    for metric_key in METRICS:
        for layout_name, run_ids in LAYOUTS.items():
            fig, ax = plt.subplots(figsize=(7, 4.5))
            plot_single_layout(ax, layout_name, run_ids, metric_key, api, cache)
            fig.tight_layout()

            slug = layout_name.lower().replace(" ", "_")
            filename = f"beta_comparison_{slug}_{metric_key}.png"
            fig.savefig(output_dir / filename)
            plt.close(fig)
            print(f"saved {output_dir / filename}")

    # Combined 3-panel figure for base_return (soups delivered)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=False)
    for ax, (layout_name, run_ids) in zip(axes, LAYOUTS.items()):
        plot_single_layout(ax, layout_name, run_ids, "base_return", api, cache)

    fig.tight_layout(w_pad=2.5)
    combined_path = output_dir / "beta_comparison_all_layouts.png"
    fig.savefig(combined_path)
    plt.close(fig)
    print(f"saved {combined_path}")


if __name__ == "__main__":
    main()
