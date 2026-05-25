"""Download xp_score_matrix.csv from W&B for each condition.

For runs whose in-process eval ran BOTH XP and XP_NO_OP passes, the W&B-uploaded
CSV is the XP_NO_OP overwrite (both passes save to the same filename). This
script detects that case (diagonal mean > 0.9 with bimodal off-diag) and warns;
for those you need to copy the XP/ matrix manually from the box at
    <hydra-dir>/xp_results/xp_score_matrix.csv
(not /xp_no_op/xp_results/...).

Standalone XP runs (e.g. reconstruction-based) and in-process runs that
completed only the XP pass before crashing have the correct CSV on W&B.

Usage:
    uv run --no-project --with wandb --with numpy --with pandas \\
        python evaluation/card_game/fetch_xp_matrices.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

MATRIX_DIR = Path(__file__).parent / "xp_matrices"
PROJECT = "g-benintendi-university-of-brescia/aht-benchmark"

# (W&B run id, target csv filename, comment)
RUNS = [
    ("6d93i8mc", "op_only.csv",          "OP only 48s"),
    ("mpzaww7r", "op_ja.csv",            "OP+JA noshape 48s"),
    ("do36ppqt", "op_ja_shaping.csv",    "OP+JA shaped 48s (standalone)"),
    ("bota0h7c", "op_comm_noshape.csv",  "OP+comm noshape 48s"),
    ("p2p1x7ez", "op_comm_match.csv",    "OP+comm match-only 48s (may be pending)"),
    ("p94s92du", "op_comm_shaping.csv",  "OP+comm shaped 48s"),
]


def parse_first_table(path: Path) -> np.ndarray:
    """First (mean) table only — same logic as xp_stats.parse_xp_matrix."""
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


def looks_like_no_op_overwrite(mat: np.ndarray) -> bool:
    """Heuristic: diagonal at ~1.0 is the giveaway — when OP wrappers are off at eval,
    deterministic policies match themselves perfectly. The proper XP pass (OP on at
    eval) has SP at training-time levels, not 1.0."""
    return float(np.diag(mat).mean()) > 0.95


def main() -> None:
    try:
        import wandb
    except ImportError:
        sys.exit("install wandb: uv pip install wandb")

    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    for rid, fname, comment in RUNS:
        try:
            r = api.run(f"{PROJECT}/{rid}")
        except Exception as e:
            print(f"  [{rid}] {fname:24s} ERROR: {e!r}")
            continue

        files = [f for f in r.files() if f.name == "xp_score_matrix.csv"]
        if not files:
            print(f"  [{rid}] {fname:24s} no xp_score_matrix.csv yet  ({comment})")
            continue

        tmp = MATRIX_DIR / f"_dl_{rid}"
        tmp.mkdir(exist_ok=True)
        try:
            files[0].download(root=str(tmp), replace=True)
            src = tmp / "xp_score_matrix.csv"
            dst = MATRIX_DIR / fname
            shutil.move(str(src), str(dst))
        finally:
            for leftover in tmp.glob("*"):
                leftover.unlink()
            tmp.rmdir()

        mat = parse_first_table(dst)
        N = mat.shape[0]
        diag_mean = float(np.diag(mat).mean())
        off_mean = float(mat[~np.eye(N, dtype=bool)].mean())
        warn = " ⚠ XP_NO_OP overwrite — replace from disk!" if looks_like_no_op_overwrite(mat) else ""
        print(f"  [{rid}] {fname:24s} N={N:>2}  diag={diag_mean:.3f}  "
              f"off={off_mean:.3f}{warn}  ({comment})")


if __name__ == "__main__":
    main()
