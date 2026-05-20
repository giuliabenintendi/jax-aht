"""XP mean and OP-vs-X significance via the delete-one-seed jackknife.

The XP mean uses ALL cross-play pairs: the mean of every off-diagonal cell of
the N x N matrix (m[i,j] = score with seed i as agent 0, seed j as agent 1,
i != j) — the role-symmetric average over all pairs 0-1, 0-2, ..., 0-(N-1),
1-2, ...

A naive t-test on those N(N-1) cells is invalid: each seed appears in 2(N-1)
of them, so the cells are not independent (pseudoreplication) and the SE comes
out far too small. The XP mean is a degree-2 U-statistic over the N seeds; the
textbook standard-error estimator is the delete-one-seed jackknife — drop
seed k (its whole row AND column), recompute the all-pairs mean -> theta_(-k),
for every seed; the spread of the N leave-one-out means gives the SE. This
uses all the pairwise data while keeping the seed as the unit of replication.

OP-vs-X significance: Welch t-test on the two conditions' XP means using their
jackknife SEs (Welch-Satterthwaite df).

Usage:
    uv run --no-project --with numpy --with scipy \
        python evaluation/card_game/xp_stats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

MATRIX_DIR = Path(__file__).parent / "xp_matrices"

CONDITIONS = [
    ("OP only", "op_only.csv"),
    ("OP + JA", "op_ja.csv"),
    ("OP + JA + shaping", "op_ja_shaping.csv"),
    ("OP + comm", "op_comm.csv"),
]
BASELINE = "OP only"


def parse_xp_matrix(path: Path) -> np.ndarray:
    """Read the `episode_return_mean` block of an xp_score_matrix.csv into an NxN array."""
    rows: list[list[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            if rows:
                break
            continue
        parts = line.split(",")
        if parts[0].startswith("episode_return"):
            if rows:
                break
            continue
        if parts[0].startswith("seed_"):
            rows.append([float(x) for x in parts[1:]])
    return np.array(rows)


def offdiag_mean(mat: np.ndarray) -> float:
    """XP mean = average over every off-diagonal cell (all cross-play pairs)."""
    n = mat.shape[0]
    return float(mat[~np.eye(n, dtype=bool)].mean())


def jackknife_xp(mat: np.ndarray) -> tuple[float, float, int]:
    """All-pairs XP mean and its delete-one-seed jackknife SE.

    Returns (xp_mean, jackknife_sem, n_seeds).
    """
    n = mat.shape[0]
    theta = offdiag_mean(mat)
    loo = np.empty(n)
    keep = np.ones(n, dtype=bool)
    for k in range(n):
        keep[k] = False
        loo[k] = offdiag_mean(mat[np.ix_(keep, keep)])
        keep[k] = True
    sem = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return theta, sem, n


def fmt_p(p: float) -> str:
    """Format a p-value: 4 decimals, or scientific notation when very small."""
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}"


def main() -> None:
    data: dict[str, tuple[float, float, int]] = {}
    for label, fname in CONDITIONS:
        path = MATRIX_DIR / fname
        if path.exists():
            data[label] = jackknife_xp(parse_xp_matrix(path))

    if BASELINE not in data:
        return
    tb, sb, nb = data[BASELINE]
    for label, (to, so, no) in data.items():
        if label == BASELINE:
            continue
        se = np.sqrt(so ** 2 + sb ** 2)
        t = (to - tb) / se
        df = se ** 4 / (so ** 4 / (no - 1) + sb ** 4 / (nb - 1))
        p_two = 2 * stats.t.sf(abs(t), df)
        p_greater = stats.t.sf(t, df)
        print(f"{label} vs {BASELINE}:  "
              f"p(two-sided)={fmt_p(p_two)}   p(one-sided/greater)={fmt_p(p_greater)}")


if __name__ == "__main__":
    main()
