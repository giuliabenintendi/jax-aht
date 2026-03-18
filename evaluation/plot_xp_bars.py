"""Grouped bar chart: SP vs XP for beta=0 and best beta, all layouts.

Matches the style from the reference image: blue solid/hatched, orange solid/hatched,
red dotted benchmark line, legend below.

Usage:
    uv run python -m evaluation.plot_xp_bars --output-dir plots/
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

# XP greedy benchmark from Forkel et al. 2026 (read from Figure 3, best alpha)
HIGH_ENTROPY_XP_BENCHMARK = {
    "Cramped Room": 220,
    "Coord Ring": 315,
    "Forced Coord": 205,
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
    """Compute per-seed SP and XP, filtering out NaN seeds.

    If sp_threshold is set, also drop seeds with SP below that value.
    """
    n = matrix.shape[0]
    sp_all = np.diag(matrix)
    xp_all = np.array([
        np.nanmean([matrix[i, j] for j in range(n) if j != i])
        for i in range(n)
    ])
    valid = ~np.isnan(sp_all)
    if sp_threshold is not None:
        valid = valid & (sp_all >= sp_threshold)
    if valid.sum() < n:
        print(f"    Dropped {n - valid.sum()} collapsed/NaN seed(s)")
    return sp_all[valid], xp_all[valid]


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
                score_matrix = parse_mean_matrix(fh.read())
            sp_score, xp_score = compute_sp_xp(score_matrix, sp_threshold=1.0)

            f = run.file("xp_jsd_matrix.csv")
            f.download(replace=True, root="/tmp/xp_bars")
            with open("/tmp/xp_bars/xp_jsd_matrix.csv") as fh:
                jsd_matrix = parse_mean_matrix(fh.read())
            sp_jsd, xp_jsd = compute_sp_xp(jsd_matrix)

            data[layout][beta] = {
                "sp": sp_score, "xp": xp_score,
                "sp_jsd": sp_jsd, "xp_jsd": xp_jsd,
            }
            print(f"  {layout} β={beta}: SP={sp_score.mean():.1f}  XP={xp_score.mean():.1f}  "
                  f"SP_JSD={sp_jsd.mean():.3f}  XP_JSD={xp_jsd.mean():.3f}")

    layouts = list(CONFIGS.keys())
    n_layouts = len(layouts)
    bar_width = 0.18
    x = np.arange(n_layouts)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Default matplotlib tab colors (matching beta comparison plot)
    blue = "C0"
    orange = "C1"

    sp_b0_means, sp_b0_sems = [], []
    xp_b0_means, xp_b0_sems = [], []
    sp_best_means, sp_best_sems = [], []
    xp_best_means, xp_best_sems = [], []
    best_beta_labels = []

    for layout in layouts:
        beta_vals = sorted(CONFIGS[layout].keys())
        b0 = beta_vals[0]
        b_best = beta_vals[1]
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

    offsets = [-1.5, -0.5, 0.5, 1.5]

    # β=0 SP (solid blue)
    ax.bar(x + offsets[0] * bar_width, sp_b0_means, bar_width, yerr=sp_b0_sems,
           color=blue, edgecolor=blue, capsize=4, error_kw={"linewidth": 1.0, "color": "black"})
    # β=0 XP (hatched blue)
    ax.bar(x + offsets[1] * bar_width, xp_b0_means, bar_width, yerr=xp_b0_sems,
           color="white", edgecolor=blue, hatch="//", linewidth=1.2,
           capsize=4, error_kw={"linewidth": 1.0, "color": "black"})
    # Best β SP (solid orange)
    ax.bar(x + offsets[2] * bar_width, sp_best_means, bar_width, yerr=sp_best_sems,
           color=orange, edgecolor=orange, capsize=4, error_kw={"linewidth": 1.0, "color": "black"})
    # Best β XP (hatched orange)
    ax.bar(x + offsets[3] * bar_width, xp_best_means, bar_width, yerr=xp_best_sems,
           color="white", edgecolor=orange, hatch="//", linewidth=1.2,
           capsize=4, error_kw={"linewidth": 1.0, "color": "black"})

    # Red dotted benchmark line per layout
    for i, layout in enumerate(layouts):
        benchmark = HIGH_ENTROPY_XP_BENCHMARK[layout]
        ax.plot([i - 2 * bar_width, i + 2 * bar_width], [benchmark, benchmark],
                color="red", linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(layouts, fontsize=11)
    ax.set_ylabel("Base Return", fontsize=12)
    ax.set_ylim(bottom=0)

    # Light horizontal grid
    ax.yaxis.grid(True, linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    # Legend upper right inside plot
    legend_handles = [
        mpatches.Patch(facecolor=blue, edgecolor=blue, label=r"$\beta$=0  SP"),
        mpatches.Patch(facecolor="white", edgecolor=blue, hatch="//", label=r"$\beta$=0  XP"),
        mpatches.Patch(facecolor=orange, edgecolor=orange, label=r"Best $\beta$  SP"),
        mpatches.Patch(facecolor="white", edgecolor=orange, hatch="//", label=r"Best $\beta$  XP"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2, alpha=0.8,
               label="High entropy XP (Forkel et al.)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="none")

    fig.tight_layout()

    path = output_dir / "xp_sp_bars.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # JSD bar chart — same layout
    fig, ax = plt.subplots(figsize=(10, 6))

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

    # log(2) reference line
    ax.axhline(np.log(2), color="red", linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(layouts, fontsize=11)
    ax.set_ylabel("JSD", fontsize=12)

    ax.yaxis.grid(True, linestyle="-", alpha=0.2)
    ax.set_axisbelow(True)

    legend_handles = [
        (mpatches.Patch(facecolor=blue, edgecolor=blue),
         mpatches.Patch(facecolor="white", edgecolor=blue, hatch="//")),
        (mpatches.Patch(facecolor=orange, edgecolor=orange),
         mpatches.Patch(facecolor="white", edgecolor=orange, hatch="//")),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2, alpha=0.8),
    ]
    legend_labels = [
        r"$\beta$=0  (SP / XP)",
        r"Best $\beta$  (SP / XP)",
        r"$\log(2)$",
    ]
    from matplotlib.legend_handler import HandlerTuple
    ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="none",
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})

    fig.tight_layout()
    path = output_dir / "xp_sp_jsd_bars.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
