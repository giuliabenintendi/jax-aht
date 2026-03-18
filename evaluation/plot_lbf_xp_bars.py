"""Bar chart: SP vs XP for LBF 3-food and 10-food, beta=0 vs beta=0.001.
Same style as the Overcooked XP bar chart.

Usage:
    uv run python -m evaluation.plot_lbf_xp_bars --output-dir plots/
"""
import argparse
import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import wandb

ENTITY = "g-benintendi-university-of-brescia"
PROJECT = "aht-benchmark"

CONFIGS = {
    "LBF 3-food": {
        0.0: "h7s03wf9",
        0.001: "o5uppyp1",
    },
    "LBF 10-food": {
        0.0: "ge7o8out",
        0.001: "lobedffe",
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
            val = row[j + 1].strip()
            matrix[i, j] = float(val) if val != "nan" else np.nan
    return matrix


def compute_sp_xp(matrix, sp_threshold=None):
    """Filter out collapsed/NaN seeds from both rows AND columns, then compute SP/XP."""
    n = matrix.shape[0]
    sp_all = np.diag(matrix)
    valid = ~np.isnan(sp_all)
    if sp_threshold is not None:
        valid = valid & (sp_all >= sp_threshold)
    if valid.sum() < n:
        print(f"    Dropped {n - valid.sum()} collapsed/NaN seed(s)")

    # Filter both rows and columns
    valid_idx = np.where(valid)[0]
    filtered = matrix[np.ix_(valid_idx, valid_idx)]
    m = filtered.shape[0]

    sp = np.diag(filtered)
    xp = np.array([
        np.nanmean([filtered[i, j] for j in range(m) if j != i])
        for i in range(m)
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

    data = {}
    for layout, betas in CONFIGS.items():
        data[layout] = {}
        for beta, run_id in betas.items():
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

            f = run.file("xp_score_matrix.csv")
            f.download(replace=True, root="/tmp/lbf_xp_bars")
            with open("/tmp/lbf_xp_bars/xp_score_matrix.csv") as fh:
                score_matrix = parse_mean_matrix(fh.read())
            sp_score, xp_score = compute_sp_xp(score_matrix)

            f = run.file("xp_jsd_matrix.csv")
            f.download(replace=True, root="/tmp/lbf_xp_bars")
            with open("/tmp/lbf_xp_bars/xp_jsd_matrix.csv") as fh:
                jsd_matrix = parse_mean_matrix(fh.read())
            sp_jsd, xp_jsd = compute_sp_xp(jsd_matrix)

            data[layout][beta] = {
                "sp": sp_score, "xp": xp_score,
                "sp_jsd": sp_jsd, "xp_jsd": xp_jsd,
            }
            print(f"  {layout} β={beta}: SP={sp_score.mean():.3f}  XP={xp_score.mean():.3f}  "
                  f"SP_JSD={sp_jsd.mean():.3f}  XP_JSD={xp_jsd.mean():.3f}")

    layouts = list(CONFIGS.keys())
    n_layouts = len(layouts)
    bar_width = 0.18
    x = np.arange(n_layouts)

    blue = "C0"
    orange = "C1"

    # Score bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    sp_b0_means, sp_b0_sems = [], []
    xp_b0_means, xp_b0_sems = [], []
    sp_best_means, sp_best_sems = [], []
    xp_best_means, xp_best_sems = [], []

    for layout in layouts:
        beta_vals = sorted(CONFIGS[layout].keys())
        b0, b_best = beta_vals[0], beta_vals[1]

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

    offsets = [-1.5, -0.5, 0.5, 1.5]

    ax.bar(x + offsets[0] * bar_width, sp_b0_means, bar_width, yerr=sp_b0_sems,
           color=blue, edgecolor=blue, capsize=4, error_kw={"linewidth": 1.0, "color": "black"})
    ax.bar(x + offsets[1] * bar_width, xp_b0_means, bar_width, yerr=xp_b0_sems,
           color="white", edgecolor=blue, hatch="//", linewidth=1.2,
           capsize=4, error_kw={"linewidth": 1.0, "color": "black"})
    ax.bar(x + offsets[2] * bar_width, sp_best_means, bar_width, yerr=sp_best_sems,
           color=orange, edgecolor=orange, capsize=4, error_kw={"linewidth": 1.0, "color": "black"})
    ax.bar(x + offsets[3] * bar_width, xp_best_means, bar_width, yerr=xp_best_sems,
           color="white", edgecolor=orange, hatch="//", linewidth=1.2,
           capsize=4, error_kw={"linewidth": 1.0, "color": "black"})

    ax.set_xticks(x)
    ax.set_xticklabels(layouts, fontsize=11)
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_ylim(bottom=0)

    ax.yaxis.grid(True, linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(facecolor=blue, edgecolor=blue, label=r"$\beta$=0  SP"),
        mpatches.Patch(facecolor="white", edgecolor=blue, hatch="//", label=r"$\beta$=0  XP"),
        mpatches.Patch(facecolor=orange, edgecolor=orange, label=r"$\beta$=0.001  SP"),
        mpatches.Patch(facecolor="white", edgecolor=orange, hatch="//", label=r"$\beta$=0.001  XP"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="none")

    fig.tight_layout()
    path = output_dir / "lbf_xp_sp_bars.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # JSD bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    jsd_sp_b0_means, jsd_sp_b0_sems = [], []
    jsd_xp_b0_means, jsd_xp_b0_sems = [], []
    jsd_sp_best_means, jsd_sp_best_sems = [], []
    jsd_xp_best_means, jsd_xp_best_sems = [], []

    for layout in layouts:
        beta_vals = sorted(CONFIGS[layout].keys())
        b0, b_best = beta_vals[0], beta_vals[1]

        d0 = data[layout][b0]
        db = data[layout][b_best]

        jsd_sp_b0_means.append(d0["sp_jsd"].mean())
        jsd_sp_b0_sems.append(d0["sp_jsd"].std() / np.sqrt(len(d0["sp_jsd"])))
        jsd_xp_b0_means.append(d0["xp_jsd"].mean())
        jsd_xp_b0_sems.append(d0["xp_jsd"].std() / np.sqrt(len(d0["xp_jsd"])))

        jsd_sp_best_means.append(db["sp_jsd"].mean())
        jsd_sp_best_sems.append(db["sp_jsd"].std() / np.sqrt(len(db["sp_jsd"])))
        jsd_xp_best_means.append(db["xp_jsd"].mean())
        jsd_xp_best_sems.append(db["xp_jsd"].std() / np.sqrt(len(db["xp_jsd"])))

    ax.bar(x + offsets[0] * bar_width, jsd_sp_b0_means, bar_width, yerr=jsd_sp_b0_sems,
           color=blue, edgecolor=blue, capsize=4, error_kw={"linewidth": 1.0, "color": "black"})
    ax.bar(x + offsets[1] * bar_width, jsd_xp_b0_means, bar_width, yerr=jsd_xp_b0_sems,
           color="white", edgecolor=blue, hatch="//", linewidth=1.2,
           capsize=4, error_kw={"linewidth": 1.0, "color": "black"})
    ax.bar(x + offsets[2] * bar_width, jsd_sp_best_means, bar_width, yerr=jsd_sp_best_sems,
           color=orange, edgecolor=orange, capsize=4, error_kw={"linewidth": 1.0, "color": "black"})
    ax.bar(x + offsets[3] * bar_width, jsd_xp_best_means, bar_width, yerr=jsd_xp_best_sems,
           color="white", edgecolor=orange, hatch="//", linewidth=1.2,
           capsize=4, error_kw={"linewidth": 1.0, "color": "black"})

    ax.axhline(np.log(2), color="red", linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(layouts, fontsize=11)
    ax.set_ylabel("JSD", fontsize=12)

    ax.yaxis.grid(True, linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(facecolor=blue, edgecolor=blue, label=r"$\beta$=0  SP"),
        mpatches.Patch(facecolor="white", edgecolor=blue, hatch="//", label=r"$\beta$=0  XP"),
        mpatches.Patch(facecolor=orange, edgecolor=orange, label=r"$\beta$=0.001  SP"),
        mpatches.Patch(facecolor="white", edgecolor=orange, hatch="//", label=r"$\beta$=0.001  XP"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2, alpha=0.8,
               label=r"$\log(2)$"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="none")

    fig.tight_layout()
    path = output_dir / "lbf_xp_sp_jsd_bars.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
