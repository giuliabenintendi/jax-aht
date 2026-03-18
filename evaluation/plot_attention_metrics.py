"""Plot attention stasis and object coverage vs beta for all layouts.

Reads attention_metrics.json files from saved checkpoints.
Produces 4 plots: stasis agent_0, stasis agent_1, coverage agent_0, coverage agent_1.

Usage:
    uv run python -m evaluation.plot_attention_metrics --output-dir plots/
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 5-seed beta sweep checkpoints (same as run_attention_eval.sh)
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


def load_metrics(base_path):
    """Load attention_metrics.json from a checkpoint directory."""
    json_path = os.path.join(base_path, "attention_metrics.json")
    if not os.path.exists(json_path):
        print(f"  MISSING: {json_path}")
        return None
    with open(json_path) as f:
        return json.load(f)


def extract_per_seed(metrics, agent, metric_type):
    """Extract per-seed values for a given agent and metric.

    metric_type: 'stasis_mean' or 'pct_objects_mean'
    Returns array of per-seed values.
    """
    per_seed = metrics.get("per_seed", [])
    values = []
    for seed_data in per_seed:
        agent_data = seed_data.get(agent, {})
        val = agent_data.get(metric_type)
        if val is not None and not np.isnan(val):
            values.append(val)
    return np.array(values)


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

    # 4 plots: stasis/coverage x agent_0/agent_1
    for metric_type, ylabel in [
        ("stasis_mean", "Attention Stasis (JSD)"),
        ("pct_objects_mean", "% Attention on Objects"),
    ]:
        for agent in ["agent_0", "agent_1"]:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            for ax, layout in zip(axes, layout_names):
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

                ax.plot(BETAS, means, "o-", color="C0", linewidth=1.5, markersize=7)
                ax.fill_between(BETAS, means - sems, means + sems, color="C0", alpha=0.2)
                ax.set_xlabel(r"$\beta$")
                ax.set_ylabel(ylabel)
                ax.set_title(layout)
                ax.set_xticks(BETAS)

            agent_label = "Agent 0" if agent == "agent_0" else "Agent 1"
            fig.suptitle(f"{ylabel} — {agent_label}", fontsize=12)
            fig.tight_layout()

            metric_slug = "stasis" if "stasis" in metric_type else "coverage"
            path = output_dir / f"attn_{metric_slug}_{agent}.png"
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
