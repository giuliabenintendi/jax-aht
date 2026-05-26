"""Aggregate Hanabi XP-matrix CSVs across conditions to bare means + bar chart.

Reads one or more `xp_score_matrix.csv` files (produced by
`evaluation/run_xp_seeds.py` per condition) and reports the mean SP-diagonal
and mean XP-off-diagonal score per condition, plus a side-by-side bar chart
saved to PNG. Deliberately minimal: no confidence intervals, no significance
tests — per the Hanabi research scope, the headline number is the bare mean.

Per-condition CSV inputs are the standard `xp_results/xp_score_matrix.csv`
output of `run_xp_evaluation`. Each is an NxN matrix of mean episode
returns indexed by (agent_0 seed, agent_1 seed); the diagonal is self-play
and the off-diagonal is cross-play.

Usage:
    uv run python -m evaluation.hanabi.aggregate_xp \\
        --condition OP    results/.../xp_results/xp_score_matrix.csv \\
        --condition OP+JA results/.../xp_results/xp_score_matrix.csv \\
        --out artifacts/hanabi_xp_compare.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_xp_score_csv(csv_path: Path) -> np.ndarray:
    """Parse the NxN mean matrix from an `xp_score_matrix.csv` file.

    The CSV is written by `save_xp_csv` as two blocks (mean, then std)
    separated by a blank row. Each block has a one-row header then N rows
    of `seed_i, m[i,0], m[i,1], ...`. We only read the mean block.
    """
    with csv_path.open() as f:
        rows = list(csv.reader(f))

    # Find the first non-empty header row (the "<label>_mean" block).
    header_idx = next(i for i, r in enumerate(rows) if r and r[0].endswith("_mean"))
    data_rows: list[list[str]] = []
    for r in rows[header_idx + 1 :]:
        if not r or not r[0].startswith("seed_"):
            break
        data_rows.append(r)
    n = len(data_rows)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, row in enumerate(data_rows):
        for j in range(n):
            matrix[i, j] = float(row[1 + j])
    return matrix


def sp_xp_means(matrix: np.ndarray) -> tuple[float, float]:
    """Return (mean of diagonal, mean of off-diagonal) — i.e. (SP, XP)."""
    n = matrix.shape[0]
    diag = np.diag(matrix)
    off = matrix[~np.eye(n, dtype=bool)]
    return float(diag.mean()), float(off.mean())


def write_bar_chart(
    labels: list[str], sp_means: list[float], xp_means: list[float], out_path: Path
) -> None:
    """Side-by-side bar chart: SP and XP score per condition."""
    fig, ax = plt.subplots(figsize=(2 + 1.5 * len(labels), 4.5))
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, sp_means, width, label="SP (diagonal)", color="#4477aa")
    ax.bar(x + width / 2, xp_means, width, label="XP (off-diagonal)", color="#ee6677")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean episode return")
    ax.set_title("Hanabi XP comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (sp, xp) in enumerate(zip(sp_means, xp_means)):
        ax.text(i - width / 2, sp, f"{sp:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, xp, f"{xp:.2f}", ha="center", va="bottom", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        nargs=2,
        action="append",
        metavar=("LABEL", "CSV"),
        required=True,
        help="Condition label and path to its xp_score_matrix.csv. Repeatable.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/hanabi_xp_compare.png"),
        help="Output PNG for the bar chart.",
    )
    args = parser.parse_args()

    labels: list[str] = []
    sp_means: list[float] = []
    xp_means: list[float] = []
    print(f"{'condition':<24} {'SP mean':>10} {'XP mean':>10}")
    print("-" * 46)
    for label, csv_path_str in args.condition:
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        matrix = parse_xp_score_csv(csv_path)
        sp, xp = sp_xp_means(matrix)
        labels.append(label)
        sp_means.append(sp)
        xp_means.append(xp)
        print(f"{label:<24} {sp:>10.4f} {xp:>10.4f}")

    write_bar_chart(labels, sp_means, xp_means, args.out)
    print(f"\nbar chart written: {args.out}")


if __name__ == "__main__":
    main()
