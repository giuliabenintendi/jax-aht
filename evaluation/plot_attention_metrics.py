"""Plot attention stasis and object coverage vs beta for all layouts.

Both agents on the same plot with different colors.
Produces individual plots + a combined 2x3 panel figure.

Usage:
    uv run python -m evaluation.plot_attention_metrics --output-dir plots/
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 5-seed beta sweep checkpoints
CHECKPOINTS = {
    "Cramped Room": {
        0.0: "results/overcooked-v1/cramped_room/ja_ippo/beta_sweep/2026-03-11_16-28-18",
        0.1: "results/overcooked-v1/cramped_room/ja_ippo/beta_sweep/2026-03-12_02-19-02",
        0.25: "results/overcooked-v1/cramped_room/ja_ippo/beta_sweep/2026-03-12_12-02-26",
        0.5: "results/overcooked-v1/cramped_room/ja_ippo/beta_sweep/2026-03-12_22-28-29",
        1.0: "results/overcooked-v1/cramped_room/ja_ippo/beta_sweep/2026-03-13_10-43-54",
    },
    "Coord Ring": {
        0.0: "results/overcooked-v1/coord_ring/ja_ippo/beta_sweep/2026-03-11_22-21-45",
        0.1: "results/overcooked-v1/coord_ring/ja_ippo/beta_sweep/2026-03-12_09-05-56",
        0.25: "results/overcooked-v1/coord_ring/ja_ippo/beta_sweep/2026-03-12_19-57-49",
        0.5: "results/overcooked-v1/coord_ring/ja_ippo/beta_sweep/2026-03-13_06-54-52",
        1.0: "results/overcooked-v1/coord_ring/ja_ippo/beta_sweep/2026-03-13_17-35-15",
    },
    "Forced Coord": {
        0.0: "results/overcooked-v1/forced_coord/ja_ippo/beta_sweep/2026-03-11_22-21-40",
        0.1: "results/overcooked-v1/forced_coord/ja_ippo/beta_sweep/2026-03-12_08-59-05",
        0.25: "results/overcooked-v1/forced_coord/ja_ippo/beta_sweep/2026-03-12_19-46-01",
        0.5: "results/overcooked-v1/forced_coord/ja_ippo/beta_sweep/2026-03-13_06-32-41",
        1.0: "results/overcooked-v1/forced_coord/ja_ippo/beta_sweep/2026-03-13_17-17-24",
    },
}

BETAS = [0.0, 0.1, 0.25, 0.5, 1.0]

AGENT_LABELS = {"agent_0": "Agent 0", "agent_1": "Agent 1"}

METRICS_INFO = {
    "stasis_mean": {
        "ylabel": "Attention Stasis (JSD)",
        "slug": "stasis",
        "colors": {"agent_0": "C3", "agent_1": "C4"},  # red, purple
    },
    "pct_objects_mean": {
        "ylabel": "% Attention on Objects",
        "slug": "coverage",
        "colors": {"agent_0": "C0", "agent_1": "C2"},  # blue, green
    },
}


def load_metrics(base_path):
    json_path = os.path.join(base_path, "attention_metrics.json")
    if not os.path.exists(json_path):
        print(f"  MISSING: {json_path}")
        return None
    with open(json_path) as f:
        return json.load(f)


def extract_per_seed(metrics, agent, metric_type):
    """Extract per-seed values. Returns array of per-seed values."""
    per_seed = metrics.get("per_seed", [])
    values = []
    for seed_data in per_seed:
        agent_data = seed_data.get(agent, {})
        val = agent_data.get(metric_type)
        if val is not None and not np.isnan(val):
            values.append(val)
    return np.array(values)


AGENT_JITTER = {"agent_0": -0.012, "agent_1": 0.012}

def plot_metric_on_ax(ax, data, layout, metric_type, colors):
    """Plot both agents on a single axis for one layout and metric."""
    for agent in ["agent_0", "agent_1"]:
        means = []
        sems = []
        for beta in BETAS:
            if beta not in data.get(layout, {}):
                means.append(np.nan)
                sems.append(0)
                continue
            vals = extract_per_seed(data[layout][beta], agent, metric_type)
            if len(vals) == 0:
                means.append(np.nan)
                sems.append(0)
            else:
                means.append(vals.mean())
                sems.append(vals.std() / np.sqrt(len(vals)))

        means = np.array(means)
        sems = np.array(sems)
        color = colors[agent]
        jitter = AGENT_JITTER[agent]
        x = np.array(BETAS) + jitter

        ax.plot(x, means, "o-", color=color, linewidth=2.0, markersize=8,
                label=AGENT_LABELS[agent])
        ax.fill_between(x, means - sems, means + sems, color=color, alpha=0.15)

    ax.set_xlabel(r"$\beta$")
    ax.set_xticks(BETAS)
    ax.set_title(layout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_names = list(CHECKPOINTS.keys())

    # Load all data
    data = {}
    for layout, betas in CHECKPOINTS.items():
        data[layout] = {}
        for beta, path in betas.items():
            metrics = load_metrics(path)
            if metrics:
                data[layout][beta] = metrics
                print(f"  {layout} β={beta}: loaded")

    # Individual 3-panel plots (one per metric)
    for metric_type, info in METRICS_INFO.items():
        colors = info["colors"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for i, (ax, layout) in enumerate(zip(axes, layout_names)):
            plot_metric_on_ax(ax, data, layout, metric_type, colors)
            ax.tick_params(labelleft=True)
            if i == 0:
                ax.set_ylabel(info["ylabel"])

        # Shared legend below
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=colors["agent_0"], marker="o", linewidth=2.0),
            Line2D([0], [0], color=colors["agent_1"], marker="o", linewidth=2.0),
        ]
        fig.legend(handles, [AGENT_LABELS["agent_0"], AGENT_LABELS["agent_1"]],
                   loc="lower center", ncol=2, fontsize=13, bbox_to_anchor=(0.5, -0.05))
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)

        path = output_dir / f"attn_{info['slug']}_both_agents.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    # Combined 2x3 figure (rows: stasis, coverage; cols: layouts)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey="row")

    all_handles = []
    for row, (metric_type, info) in enumerate(METRICS_INFO.items()):
        colors = info["colors"]
        for col, layout in enumerate(layout_names):
            ax = axes[row, col]
            plot_metric_on_ax(ax, data, layout, metric_type, colors)
            ax.tick_params(labelleft=True)
            if col == 0:
                ax.set_ylabel(info["ylabel"])
            if row == 0:
                ax.set_xlabel("")

    # Use stasis colors for legend (both rows show Agent 0 / Agent 1)
    from matplotlib.lines import Line2D
    stasis_colors = METRICS_INFO["stasis_mean"]["colors"]
    coverage_colors = METRICS_INFO["pct_objects_mean"]["colors"]
    handles = [
        Line2D([0], [0], color="gray", marker="o", linewidth=2.0),
        Line2D([0], [0], color="gray", marker="o", linewidth=2.0, linestyle="--"),
    ]
    # Use actual per-row colors in the legend
    handles = [
        Line2D([0], [0], color=stasis_colors["agent_0"], marker="o", linewidth=2.0,
               label="Agent 0 (stasis: red, coverage: orange)"),
        Line2D([0], [0], color=stasis_colors["agent_1"], marker="o", linewidth=2.0,
               label="Agent 1 (stasis: green, coverage: blue)"),
    ]
    fig.legend(handles, [AGENT_LABELS["agent_0"], AGENT_LABELS["agent_1"]],
               loc="lower center", ncol=2, fontsize=13, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06)

    combined_path = output_dir / "attn_combined_2x3.png"
    fig.savefig(combined_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {combined_path}")


if __name__ == "__main__":
    main()
