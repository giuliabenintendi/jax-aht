"""XP mean / SE and OP-vs-X significance.

Two estimators are provided.

1) For PLOTTING (`xp_mean_se`):
   All-pairs XP mean (mean of every off-diagonal cell) with delete-one-seed
   resampling SE. Uses every pairwise observation while keeping the seed as
   the unit of replication — a degree-2 U-statistic SE.

2) For SIGNIFICANCE (`paired_test_vs_baseline`):
   Disjoint-pair paired t-test (m = N/2 independent samples). For each pair
   (i, j), score = (M[i,j] + M[j,i]) / 2 (symmetric over role). Paired
   t-test (one-sided 'greater') between the condition's m samples and the
   baseline's m samples.

The naive cell-level t-test on N(N-1) off-diagonal entries is invalid
(pseudoreplication: each seed appears in 2(N-1) cells). Both estimators above
respect the seed as the unit of independence.

Usage:
    uv run --no-project --with numpy --with scipy \\
        python evaluation/card_game/xp_stats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

MATRIX_DIR = Path(__file__).parent / "xp_matrices"

CONDITIONS = [
    ("OP only",                "op_only.csv"),
    ("OP + JA",                "op_ja.csv"),
    ("OP + JA + shaping",      "op_ja_shaping.csv"),
    ("OP + comm",              "op_comm_noshape.csv"),
    ("OP + comm + match",      "op_comm_match.csv"),
    ("OP + comm + shaping",    "op_comm_shaping.csv"),
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


def xp_mean_se(mat: np.ndarray) -> tuple[float, float, int]:
    """All-pairs XP mean and its delete-one-seed standard error.

    Returns (xp_mean, sem, n_seeds).
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


def disjoint_pair_samples(mat: np.ndarray) -> np.ndarray:
    """m = N/2 symmetric XP scores from disjoint pairs (0,1),(2,3),...,(N-2,N-1).

    Each pair (i, j) contributes one sample: 0.5 * (M[i,j] + M[j,i]).
    """
    n = mat.shape[0]
    if n % 2 != 0:
        raise ValueError(f"need even N for disjoint pairing, got {n}")
    i_idx = np.arange(0, n, 2)
    j_idx = np.arange(1, n, 2)
    return 0.5 * (mat[i_idx, j_idx] + mat[j_idx, i_idx])


def paired_test_vs_baseline(mat_cond: np.ndarray, mat_baseline: np.ndarray,
                             alternative: str = "greater") -> dict:
    """Paired t-test on m = N/2 disjoint-pair XP samples (condition vs baseline).

    Returns dict with: a_mean, a_std, b_mean, b_std, mean_diff, ci_lo, ci_hi,
    t, p, n.
    """
    a = disjoint_pair_samples(mat_cond)
    b = disjoint_pair_samples(mat_baseline)
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal n; got {len(a)} vs {len(b)}")
    t, p = stats.ttest_rel(a, b, alternative=alternative)
    mean_diff = float((a - b).mean())
    ci_lo, ci_hi = stats.t.interval(0.95, df=len(a) - 1,
                                    loc=mean_diff,
                                    scale=stats.sem(a - b))
    return {
        "a_mean": float(a.mean()), "a_std": float(a.std(ddof=1)),
        "b_mean": float(b.mean()), "b_std": float(b.std(ddof=1)),
        "mean_diff": mean_diff, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "t": float(t), "p": float(p), "n": int(len(a)),
    }


def fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}"


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main() -> None:
    mats: dict[str, np.ndarray] = {}
    for label, fname in CONDITIONS:
        path = MATRIX_DIR / fname
        if path.exists():
            mats[label] = parse_xp_matrix(path)
        else:
            print(f"skip (no csv): {label}  [{fname}]")

    if BASELINE not in mats:
        print(f"missing baseline ({BASELINE}); cannot run tests")
        return

    baseline_mat = mats[BASELINE]

    print(f"\n=== Paired t-test (disjoint pairs, m = N/2, one-sided 'greater') "
          f"vs {BASELINE} ===")
    print(f"{'condition':<24s} {'XP':>6s} {'Δ':>7s} {'95% CI':>20s} "
          f"{'t':>6s} {'p':>10s} {'sig':>5s} {'n':>3s}")
    for label, mat in mats.items():
        if label == BASELINE:
            continue
        if mat.shape != baseline_mat.shape:
            print(f"  {label}: shape {mat.shape} != baseline {baseline_mat.shape}, skipping")
            continue
        r = paired_test_vs_baseline(mat, baseline_mat, alternative="greater")
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(f"{label:<24s} {r['a_mean']:>6.3f} {r['mean_diff']:>+7.3f} {ci:>20s} "
              f"{r['t']:>6.2f} {fmt_p(r['p']):>10s} {stars(r['p']):>5s} {r['n']:>3d}")


if __name__ == "__main__":
    main()
