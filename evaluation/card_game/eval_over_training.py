"""Evaluate SP/XP at every (subsampled) training checkpoint for one condition.

For each selected per-chunk checkpoint folder under `--ckpt-root`:
  - Load the chunk's stacked-seed params as pure numpy (device-agnostic).
  - Write a temporary `saved_train_run/` under <hydra-dir>/chunk_eval/chunk_NN/
    (with `.hydra/config.yaml` mirrored so run_xp_evaluation finds the algo cfg).
  - Run the standard `evaluation.run_xp_seeds.run_xp_evaluation` (W&B disabled
    via env var) → produces `xp_results/xp_score_matrix.csv` next to it.
  - Parse the matrix; record diag-mean/std (SP) and off-diag-mean/std (XP).

Output: a CSV with one row per evaluated chunk
  columns: condition, chunk, env_step, sp_mean, sp_std, xp_mean, xp_std

Run from the repo root:
    ./run_gpu.sh <GPU> evaluation.card_game.eval_over_training \\
        --ckpt-root /scratch/.../checkpoints/<run-name>/ \\
        --hydra-dir /scratch/.../results/.../<timestamp>/ \\
        --label "OP+JA+shaping" \\
        --out-csv evaluation/card_game/over_training/op_ja_shaping.csv \\
        --every 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

# W&B clutter: each chunk eval would otherwise create its own run. Disable
# entirely; the per-chunk xp_score_matrix.csv is what we actually need.
os.environ.setdefault("WANDB_MODE", "disabled")

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import orbax.checkpoint as ocp
from flax.training import orbax_utils


def to_numpy_tree(obj):
    """Recursively strip jax sharding info — every leaf becomes a numpy array."""
    if isinstance(obj, dict):
        return {k: to_numpy_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_numpy_tree(x) for x in obj)
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "__array__"):
        return np.asarray(obj)
    return obj


def load_chunk_numpy(path: str):
    checkpointer = ocp.PyTreeCheckpointer()
    return to_numpy_tree(checkpointer.restore(path))


def parse_mean_matrix(csv_path: Path) -> np.ndarray:
    """First (mean) table from xp_score_matrix.csv → NxN numpy array."""
    rows: list[list[float]] = []
    for raw in csv_path.read_text().splitlines():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True,
                    help="folder containing ckpt_NN_* subdirs and chunk_scores.json")
    ap.add_argument("--hydra-dir", required=True,
                    help="original hydra output dir (must contain .hydra/config.yaml)")
    ap.add_argument("--label", required=True,
                    help="condition label for the CSV (e.g. 'OP+JA+shaping')")
    ap.add_argument("--out-csv", required=True,
                    help="output CSV path; appends if exists")
    ap.add_argument("--every", type=int, default=4,
                    help="evaluate every Nth chunk (default 4)")
    ap.add_argument("--first-n", type=int, default=None,
                    help="optional cap on number of chunks to eval")
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_root)
    hydra_dir = Path(args.hydra_dir)
    src_hydra = hydra_dir / ".hydra"
    if not (src_hydra / "config.yaml").exists():
        raise FileNotFoundError(f"no .hydra/config.yaml in {hydra_dir}")

    scores = json.loads((ckpt_root / "chunk_scores.json").read_text())
    ckpt_paths = scores["ckpt_folder_paths"]
    env_steps = scores["ckpt_env_steps"]
    num_ckpts = int(scores["num_ckpts"])

    chunk_indices = list(range(0, num_ckpts, args.every))
    if chunk_indices and chunk_indices[-1] != num_ckpts - 1:
        chunk_indices.append(num_ckpts - 1)  # always include the final chunk
    if args.first_n:
        chunk_indices = chunk_indices[:args.first_n]
    print(f"[{args.label}] evaluating {len(chunk_indices)}/{num_ckpts} chunks: {chunk_indices}")

    eval_root = hydra_dir / "chunk_eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    # Heavy import: brings JAX + the eval primitive online once
    from evaluation.run_xp_seeds import run_xp_evaluation

    rows: list[dict] = []
    for ci in chunk_indices:
        chunk_dir = eval_root / f"chunk_{ci:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        dst_hydra = chunk_dir / ".hydra"
        if not dst_hydra.exists():
            shutil.copytree(src_hydra, dst_hydra)

        print(f"  chunk {ci}: loading {ckpt_paths[ci]}")
        chunk_params = load_chunk_numpy(ckpt_paths[ci])
        out_dict = to_numpy_tree({
            "best_params": chunk_params,
            "final_params": chunk_params,
        })
        save_path = chunk_dir / "saved_train_run"
        if save_path.exists():
            shutil.rmtree(save_path)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(str(save_path), out_dict,
                          save_args=orbax_utils.save_args_from_target(out_dict))

        print(f"  chunk {ci}: running eval...")
        try:
            run_xp_evaluation(task_name=None, checkpoint_path=str(save_path),
                              greedy_eval=True, use_best=False, drop_op=False,
                              no_xp_videos=True)
        except Exception as e:
            print(f"    eval failed: {e!r}")
            continue

        matrix_csv = chunk_dir / "xp_results" / "xp_score_matrix.csv"
        if not matrix_csv.exists():
            print(f"    no xp_score_matrix.csv at {matrix_csv}")
            continue

        M = parse_mean_matrix(matrix_csv)
        N = M.shape[0]
        diag = np.diag(M)
        off = M[~np.eye(N, dtype=bool)]
        row = {
            "condition": args.label,
            "chunk": ci,
            "env_step": int(env_steps[ci]),
            "sp_mean": float(diag.mean()),
            "sp_std": float(diag.std(ddof=1)),
            "xp_mean": float(off.mean()),
            "xp_std": float(off.std(ddof=1)),
        }
        rows.append(row)
        print(f"  chunk {ci}: env_step={row['env_step']:>10}  "
              f"SP={row['sp_mean']:.3f}±{row['sp_std']:.3f}  "
              f"XP={row['xp_mean']:.3f}±{row['xp_std']:.3f}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "chunk", "env_step", "sp_mean", "sp_std", "xp_mean", "xp_std"]
    write_header = not out_csv.exists()
    mode = "w" if write_header else "a"
    with open(out_csv, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsaved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
