"""Learning curves for all conditions: training return vs env_step.

For each condition, pulls `LiveTrain/seed_{i}/return_mean` from W&B, aggregates
across seeds at each env_step (mean and across-seed SEM), and plots one line
+ shaded band per condition.

Aggregated curves are cached as evaluation/card_game/learning_curves/<label>.csv
so subsequent runs are instant. Delete a cached csv to force a refetch.

Usage:
    uv run python evaluation/card_game/plot_learning_curves.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent / "learning_curves"
OUT = Path("plots/card_game/learning_curves.png")
PROJECT = "g-benintendi-university-of-brescia/aht-benchmark"

RANDOM_COLOR = "#d62728"
RANDOM_BASELINE = 0.20

# (display label, W&B run id, color, cache filename)
ROWS = [
    ("OP only",                "6d93i8mc", "#2E7DF0", "op_only.csv"),
    ("OP + JA",                "mpzaww7r", "#F5871F", "op_ja.csv"),
    ("OP + comm",              "bota0h7c", "#F8A0AE", "op_comm_noshape.csv"),
    ("OP + JA + shaping",      "dx04ii07", "#51B18D", "op_ja_shaping.csv"),
    ("OP + comm + match",      "p2p1x7ez", "#F77F92", "op_comm_match.csv"),
    ("OP + comm + shaping",    "p94s92du", "#F55F74", "op_comm_shaping.csv"),
]

PAT = re.compile(r"LiveTrain/seed_(\d+)/return_mean$")


def fetch_and_aggregate(rid: str, cache_path: Path) -> pd.DataFrame | None:
    """Pull per-seed LiveTrain returns from W&B, aggregate per env_step.

    Returned DataFrame: env_step, mean, std, count.
    """
    if cache_path.exists():
        print(f"  cached: {cache_path.name}")
        return pd.read_csv(cache_path)
    try:
        import wandb
    except ImportError:
        print("  wandb not installed; cannot fetch")
        return None
    api = wandb.Api()
    try:
        r = api.run(f"{PROJECT}/{rid}")
    except Exception as e:
        print(f"  cannot get run {rid}: {e}")
        return None

    rows: list[tuple[int, int, float]] = []
    for row in r.scan_history():
        es = row.get("env_step")
        if es is None:
            continue
        for k, v in row.items():
            m = PAT.match(k)
            if m and v is not None:
                rows.append((int(es), int(m.group(1)), float(v)))
    if not rows:
        print(f"  no LiveTrain data for {rid}")
        return None

    df = pd.DataFrame(rows, columns=["env_step", "seed", "ret"])
    agg = (df.groupby("env_step")["ret"]
             .agg(["mean", "std", "count"])
             .reset_index()
             .sort_values("env_step"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(cache_path, index=False)
    print(f"  fetched {len(agg)} rows for {rid}  "
          f"(seeds present: 1..{agg['count'].max()})  -> {cache_path.name}")
    return agg


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))

    for label, rid, color, fname in ROWS:
        print(f"{label}  ({rid}):")
        cache_path = CACHE_DIR / fname
        agg = fetch_and_aggregate(rid, cache_path)
        if agg is None or len(agg) == 0:
            continue
        x = agg["env_step"].values
        y = agg["mean"].values
        n = agg["count"].fillna(1).values.astype(float)
        s = agg["std"].fillna(0).values
        sem = s / np.sqrt(np.maximum(n, 1.0))
        ax.plot(x, y, label=label, color=color, lw=2.0, zorder=4)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.2, zorder=3)
        print(f"  final point: env_step={x[-1]}  mean={y[-1]:.3f}  sem={sem[-1]:.4f}")

    ax.axhline(RANDOM_BASELINE, ls="--", lw=1.5, color=RANDOM_COLOR,
               label="Random baseline", zorder=2)
    ax.set_xlabel("Environment steps", fontsize=11)
    ax.set_ylabel("Training return (mean ± SEM across seeds)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
