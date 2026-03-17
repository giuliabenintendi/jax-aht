"""Grouped bar chart: SP vs XP for beta=0 and best beta, all layouts on one axis.

Usage:
    uv run python -m evaluation.plot_xp_bars --output-dir plots/
"""
import argparse
import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import wandb

ENTITY = "g-benintendi-university-of-brescia"
PROJECT = "aht-benchmark"

# {layout: {beta: xp_eval_run_id}}
CONFIGS = {
    "Cramped Room": {
        0.0: "c5mmg8an",
        0.25: "tso4qleg",
    },
    "Coord Ring": {
        0.0: "090zw12u",
        0.25: "vzsho5ht",
    },
    "Forced Coord": {
        0.0: "l3lzx4fy",
        0.5: "ykagespv",
    },
}


def parse_mean_matrix(csv_text):
    sections = csv_text.strip().split("\n\n")
    reader = csv.reader(io.StringIO(sections[0]))
    header = next(reader)
    n = len(header) - 1
    matrix = np.zeros((n, n))
    for i, row in enumerate(reader):
        for j in range(n):
            matrix[i, j] = float(row[j + 1])
    return matrix


def compute_sp_xp(score_matrix):
    n = score_matrix.shape[0]
    sp = np.diag(score_matrix)
    xp = np.array([
        np.mean([score_matrix[i, j] for j in range(n) if j != i])
        for i in range(n)
    ])
    return sp, xp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()

    # Fetch all matrices
    data = {}
    for layout, betas in CONFIGS.items():
        data[layout] = {}
        for beta, run_id in betas.items():
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
            f = run.file("xp_score_matrix.csv")
            f.download(replace=True, root="/tmp/xp_bars")
            with open("/tmp/xp_bars/xp_score_matrix.csv") as fh:
                matrix = parse_mean_matrix(fh.read())
            sp, xp = compute_sp_xp(matrix)
            data[layout][beta] = {"sp": sp, "xp": xp}
            print(f"  {layout} β={beta}: SP={sp.mean():.1f}±{sp.std()/np.sqrt(len(sp)):.1f}  "
                  f"XP={xp.mean():.1f}±{xp.std()/np.sqrt(len(xp)):.1f}")

    # Single figure: all layouts on x-axis
    # For each layout: 4 bars (SP β=0, XP β=0, SP β=best, XP β=best)
    layouts = list(CONFIGS.keys())
    n_layouts = len(layouts)
    bar_width = 0.18
    x = np.arange(n_layouts)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors: blue for β=0, orange for β=best
    # Solid for SP, hatched for XP
    colors = {"b0_sp": "C0", "b0_xp": "C0", "best_sp": "C1", "best_xp": "C1"}

    sp_b0_means, sp_b0_sems = [], []
    xp_b0_means, xp_b0_sems = [], []
    sp_best_means, sp_best_sems = [], []
    xp_best_means, xp_best_sems = [], []
    best_beta_labels = []

    for layout in layouts:
        beta_vals = sorted(CONFIGS[layout].keys())
        b0 = beta_vals[0]       # 0.0
        b_best = beta_vals[1]   # best beta
        best_beta_labels.append(f"β={b_best}")

        d0 = data[layout][b0]
        db = data[layout][b_best]

        sp_b0_means.append(d0["sp"].mean())
        sp_b0_sems.append(d0["sp"].std() / np.sqrt(len(d0["sp"])))
        xp_b0_means.append(d0["xp"].mean())
        xp_b0_sems.append(d0["xp"].std() / np.sqrt(len(d0["xp"])))

        sp_best_means.append(db["sp"].mean())
        sp_best_sems.append(db["sp"].std() / np.sqrt(len(db["sp"])))
        xp_best_means.append(db["xp"].mean())
        xp_best_sems.append(db["xp"].std() / np.sqrt(len(db["xp"])))

    # 4 groups of bars per layout
    offsets = [-1.5, -0.5, 0.5, 1.5]

    ax.bar(x + offsets[0] * bar_width, sp_b0_means, bar_width, yerr=sp_b0_sems,
           color="C0", capsize=4)
    ax.bar(x + offsets[1] * bar_width, xp_b0_means, bar_width, yerr=xp_b0_sems,
           color="C0", hatch="//", edgecolor="C0", capsize=4)
    ax.bar(x + offsets[2] * bar_width, sp_best_means, bar_width, yerr=sp_best_sems,
           color="C1", capsize=4)
    ax.bar(x + offsets[3] * bar_width, xp_best_means, bar_width, yerr=xp_best_sems,
           color="C1", hatch="//", edgecolor="C1", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(layouts)
    ax.set_ylabel("Episode Return")
    ax.set_title("Self-Play vs Cross-Play")

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="C0", label="β = 0  SP"),
        mpatches.Patch(facecolor="C0", hatch="//", edgecolor="C0", label="β = 0  XP"),
        mpatches.Patch(facecolor="C1", label="Best β  SP"),
        mpatches.Patch(facecolor="C1", hatch="//", edgecolor="C1", label="Best β  XP"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    fig.tight_layout()
    path = output_dir / "xp_sp_bars.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
