"""Plot return-vs-progress curves for the episode-length ablation, on a
normalised x-axis so cells with different TOTAL_TIMESTEPS overlay cleanly.

The ablation scales TOTAL_TIMESTEPS linearly with max_steps to fix
NUM_UPDATES across cells, so plotting against env_step puts the cells at
different terminal x-positions. This script converts env_step to one of:
  - train_step  (= env_step / (NUM_ENVS * max_steps), default)
  - episodes    (= env_step / max_steps)
  - fraction    (= env_step / TOTAL_TIMESTEPS)
which all align the cells on a common axis.

Usage:
    uv run python -m evaluation.plot_max_steps_ablation \\
        --label-prefix max_steps_ablation_ \\
        --output plots/card_game/max_steps_ablation.png

Filters wandb runs by label prefix; aggregates per-seed `LiveTrain/seed_*/return_mean`
into a mean ± across-seed std band per cell, then plots one line per max_steps.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _fetch_curves(api, entity: str, project: str, label_prefix: str):
    """Return {max_steps: dict(env_step, mean, std, num_envs, total_timesteps)}.

    wandb quirks worked around here:
      - `$regex` rejects `^anchor` but accepts `prefix.*`, so the filter is
        written that way.
      - Paginated `runs()` returns shells with empty configs (and `.load()`
        doesn't fix it), so each matched run is refetched via `api.run(id)`
        to get the full algorithm/env config and access history.
    """
    filters = {
        "config.label": {"$regex": f"{label_prefix}.*"},
        "state": {"$in": ["finished", "running"]},
    }
    shells = list(api.runs(f"{entity}/{project}", filters=filters))
    print(f"  found {len(shells)} runs matching label prefix '{label_prefix}'")

    out: dict[int, dict] = {}
    for shell in shells:
        r = api.run(f"{entity}/{project}/{shell.id}")
        alg = r.config.get("algorithm", {})
        env_kwargs = alg.get("ENV_KWARGS", {})
        max_steps = int(env_kwargs.get("max_steps", 0))
        if not max_steps:
            print(f"    [skip] {r.id}: no max_steps in config")
            continue
        n_seeds = int(alg.get("NUM_SEEDS", 1))
        num_envs = int(alg.get("NUM_ENVS", 1024))
        total_ts = int(alg.get("TOTAL_TIMESTEPS", 0))

        # Collect per-step per-seed return_mean via scan_history with forward-fill.
        by_step: dict[int, list] = {}
        for row in r.scan_history(page_size=1000):
            es = row.get("env_step")
            if es is None:
                continue
            per_seed = {}
            for k, v in row.items():
                if k.startswith("LiveTrain/seed_") and k.endswith("/return_mean") and v is not None:
                    si = int(k.split("seed_")[1].split("/")[0])
                    per_seed[si] = v
            if not per_seed:
                continue
            bucket = by_step.setdefault(es, [None] * n_seeds)
            for si, v in per_seed.items():
                bucket[si] = v
        if not by_step:
            print(f"    [skip] {r.id}: no return_mean history")
            continue

        steps = np.array(sorted(by_step.keys()))
        arr = np.array([by_step[s] for s in steps], dtype=float)
        # Forward-fill per column
        for col in range(arr.shape[1]):
            last = np.nan
            for row_i in range(arr.shape[0]):
                if np.isnan(arr[row_i, col]):
                    arr[row_i, col] = last
                else:
                    last = arr[row_i, col]
        mean = np.nanmean(arr, axis=1)
        std = np.nanstd(arr, axis=1)

        out[max_steps] = {
            "env_step": steps,
            "mean": mean,
            "std": std,
            "num_envs": num_envs,
            "total_timesteps": total_ts,
            "n_seeds": n_seeds,
            "run_id": r.id,
        }
        print(
            f"    max_steps={max_steps:>2}  run_id={r.id}  n_seeds={n_seeds}  "
            f"history_points={len(steps)}  total_ts={total_ts}  final_mean={mean[-1]:.3f}"
        )
    return out


def _x_transform(env_step: np.ndarray, info: dict, max_steps: int, mode: str) -> np.ndarray:
    if mode == "train_step":
        return env_step / max(info["num_envs"] * max_steps, 1)
    if mode == "episodes":
        return env_step / max(max_steps, 1)
    if mode == "fraction":
        return env_step / max(info["total_timesteps"], 1)
    raise ValueError(f"unknown x-axis mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="g-benintendi-university-of-brescia")
    parser.add_argument("--project", default="aht-benchmark")
    parser.add_argument(
        "--label-prefix", default="max_steps_ablation_",
        help="wandb config.label prefix to match all sweep cells",
    )
    parser.add_argument(
        "--x-axis", choices=["train_step", "episodes", "fraction"],
        default="train_step",
    )
    parser.add_argument("--output", default="plots/card_game/max_steps_ablation.png")
    parser.add_argument("--smooth", type=int, default=1,
                        help="moving-average window over the curve (1 = no smoothing)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["savefig.dpi"] = 200
    import wandb
    api = wandb.Api(timeout=120)

    print(f"Fetching runs from {args.entity}/{args.project} ...")
    curves = _fetch_curves(api, args.entity, args.project, args.label_prefix)
    if not curves:
        print("No runs found. Exiting.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    sorted_ms = sorted(curves.keys())
    for i, ms in enumerate(sorted_ms):
        info = curves[ms]
        x = _x_transform(info["env_step"], info, ms, args.x_axis)
        m = info["mean"]
        s = info["std"]
        if args.smooth > 1:
            kernel = np.ones(args.smooth) / args.smooth
            m = np.convolve(m, kernel, mode="same")
            s = np.convolve(s, kernel, mode="same")
        color = cmap(i / max(len(sorted_ms) - 1, 1))
        ax.plot(x, m, label=f"max_steps={ms} (n={info['n_seeds']})",
                color=color, linewidth=2)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.18)

    xlabel = {
        "train_step": "training updates",
        "episodes": "episodes seen",
        "fraction": "fraction of training",
    }[args.x_axis]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("episode return (mean across seeds)")
    ax.set_title("Episode-length ablation (rescaled training, fixed NUM_UPDATES)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"\nSaved {out_path.resolve()}")


if __name__ == "__main__":
    main()
