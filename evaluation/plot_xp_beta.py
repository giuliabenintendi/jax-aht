"""Plot XP/SP ratio and XP JSD across beta values for all layouts.

Computes per-seed XP/SP ratio from the full NxN score matrices,
then averages across seeds with SEM.

Usage:
    uv run python -m evaluation.plot_xp_beta --output-dir plots/
"""
import argparse
import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "g-benintendi-university-of-brescia"
PROJECT = "aht-benchmark"

# XP eval run IDs (tag=xp_eval)
LAYOUTS = {
    "Cramped Room": {
        0.0: "c5mmg8an",
        0.1: "18idvufd",
        0.25: "tso4qleg",
        0.5: "byg2y8ca",
        1.0: "gtrebxys",
    },
    "Coord Ring": {
        0.0: "090zw12u",
        0.1: "m4jndk6c",
        0.25: "vzsho5ht",
        0.5: "dcp05c5c",
        1.0: "6qnrlcdz",
    },
    "Forced Coord": {
        0.0: "l3lzx4fy",
        0.1: "sihcjauz",
        0.25: "7sd03uu7",
        0.5: "ykagespv",
        1.0: "gekvg58w",
    },
}

SP_THRESHOLD = 50


def parse_matrix_csv(csv_text):
    """Parse the NxN mean matrix from the XP CSV format."""
    sections = csv_text.strip().split("\n\n")
    # First section is the mean matrix
    reader = csv.reader(io.StringIO(sections[0]))
    header = next(reader)  # e.g. "episode_return_mean,seed_0,seed_1,..."
    n = len(header) - 1
    matrix = np.zeros((n, n))
    for i, row in enumerate(reader):
        for j in range(n):
            matrix[i, j] = float(row[j + 1])
    return matrix


def fetch_matrices(api):
    """Download score and JSD matrices for all runs."""
    data = {}
    for layout, runs in LAYOUTS.items():
        data[layout] = {}
        for beta, run_id in runs.items():
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

            # Download score matrix
            score_file = run.file("xp_score_matrix.csv")
            score_file.download(replace=True, root="/tmp/xp_csv")
            with open("/tmp/xp_csv/xp_score_matrix.csv") as f:
                score_matrix = parse_matrix_csv(f.read())

            # Download JSD matrix
            jsd_file = run.file("xp_jsd_matrix.csv")
            jsd_file.download(replace=True, root="/tmp/xp_csv")
            with open("/tmp/xp_csv/xp_jsd_matrix.csv") as f:
                jsd_matrix = parse_matrix_csv(f.read())

            data[layout][beta] = {
                "score": score_matrix,
                "jsd": jsd_matrix,
            }
            n = score_matrix.shape[0]
            sp = np.diag(score_matrix).mean()
            print(f"  {layout} β={beta}: {n}x{n} matrix, SP={sp:.1f}")
    return data


def compute_xp_sp_ratios(score_matrix):
    """Compute per-seed XP/SP ratio, return (per_seed_means, sp_values).

    For each seed i: ratio_i = mean(M[i,j] / M[i,i] for j≠i)
    """
    n = score_matrix.shape[0]
    per_seed_ratios = []
    sp_values = np.diag(score_matrix)

    for i in range(n):
        sp_i = score_matrix[i, i]
        if sp_i < 1e-6:
            per_seed_ratios.append(np.nan)
            continue
        xp_ratios = [score_matrix[i, j] / sp_i for j in range(n) if j != i]
        per_seed_ratios.append(np.mean(xp_ratios))

    return np.array(per_seed_ratios), sp_values


def compute_xp_jsd(jsd_matrix):
    """Compute per-seed mean XP JSD.

    For each seed i: jsd_i = mean(M[i,j] for j≠i)
    """
    n = jsd_matrix.shape[0]
    per_seed_jsd = []
    for i in range(n):
        xp_jsds = [jsd_matrix[i, j] for j in range(n) if j != i]
        per_seed_jsd.append(np.mean(xp_jsds))
    return np.array(per_seed_jsd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    data = fetch_matrices(api)

    betas = [0.0, 0.1, 0.25, 0.5, 1.0]
    layout_names = list(LAYOUTS.keys())

    # Plot 1: XP/SP ratio (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, layout in zip(axes, layout_names):
        means = []
        sems = []
        marker_colors = []

        for beta in betas:
            score_matrix = data[layout][beta]["score"]
            per_seed_ratios, sp_values = compute_xp_sp_ratios(score_matrix)

            valid = ~np.isnan(per_seed_ratios)
            ratios = per_seed_ratios[valid] * 100
            mean_sp = sp_values.mean()

            means.append(np.mean(ratios))
            sems.append(np.std(ratios) / np.sqrt(len(ratios)))
            marker_colors.append("C0" if mean_sp >= SP_THRESHOLD else "0.6")

        means = np.array(means)
        sems = np.array(sems)

        ax.plot(betas, means, color="C0", linewidth=1.5, zorder=2)
        ax.fill_between(betas, means - sems, means + sems, color="C0", alpha=0.25, zorder=1)
        for i, (b, m, c) in enumerate(zip(betas, means, marker_colors)):
            ax.plot(b, m, 'o', color=c, markersize=7, zorder=3)
            if c == "0.6":
                ax.annotate("SP collapsed", (b, m), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=7, color="0.5")

        ax.axhline(100, color="red", linestyle=":", linewidth=1, alpha=0.7, label="Perfect XP")
        ax.set_xlabel(r"$\beta$")
        if ax == axes[0]:
            ax.set_ylabel("XP / SP (%)")
        ax.set_title(layout)
        ax.set_xticks(betas)
        ax.legend(fontsize=8)

    fig.tight_layout(w_pad=2.0)
    path = output_dir / "xp_sp_ratio_vs_beta.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {path}")

    # Plot 2: XP JSD (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, layout in zip(axes, layout_names):
        jsd_means = []
        jsd_sems = []

        for beta in betas:
            jsd_matrix = data[layout][beta]["jsd"]
            per_seed_jsd = compute_xp_jsd(jsd_matrix)

            jsd_means.append(np.mean(per_seed_jsd))
            jsd_sems.append(np.std(per_seed_jsd) / np.sqrt(len(per_seed_jsd)))

        jsd_means = np.array(jsd_means)
        jsd_sems = np.array(jsd_sems)

        ax.plot(betas, jsd_means, color="C1", linewidth=1.5, zorder=2)
        ax.fill_between(betas, jsd_means - jsd_sems, jsd_means + jsd_sems,
                         color="C1", alpha=0.25, zorder=1)
        ax.plot(betas, jsd_means, 'o', color="C1", markersize=7, zorder=3)

        ax.axhline(np.log(2), color="red", linestyle=":", linewidth=1, alpha=0.7,
                    label=f"log(2) = {np.log(2):.3f}")
        ax.set_xlabel(r"$\beta$")
        if ax == axes[0]:
            ax.set_ylabel("XP JSD")
        ax.set_title(layout)
        ax.set_ylim(0, 0.75)
        ax.set_xticks(betas)
        ax.legend(fontsize=8)

    fig.tight_layout(w_pad=2.0)
    path = output_dir / "xp_jsd_vs_beta.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
