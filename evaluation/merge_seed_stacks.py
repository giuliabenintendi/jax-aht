"""Merge per-run orbax seed stacks into one multi-seed checkpoint.

Block runs train NUM_SEEDS=k seeds each (stacked as (k, ...) in
saved_train_run), but `run_xp_seeds.py` expects a single checkpoint whose
best_params/final_params carry ALL N seeds. This concatenates the stacks of
several runs of the SAME architecture along the seed axis and writes a
run-dir-shaped output (saved_train_run + .hydra/config.yaml copied from the
first input) that the XP pipeline can consume directly.

Usage:
    uv run python -m evaluation.merge_seed_stacks OUT_DIR RUN_DIR [RUN_DIR ...]
"""
import glob
import os
import shutil
import sys

import jax
import numpy as np
import orbax.checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.save_load_utils import load_train_run_no_convert


def _find_ckpt(run_dir):
    direct = os.path.join(run_dir, "saved_train_run")
    if os.path.isdir(direct):
        return direct
    hits = [p for p in glob.glob(os.path.join(run_dir, "*"))
            if os.path.isdir(p) and glob.glob(os.path.join(p, "*METADATA*"))]
    if not hits:
        raise FileNotFoundError(f"no orbax checkpoint under {run_dir}")
    return hits[0]


def main():
    out_dir = os.path.abspath(sys.argv[1])
    run_dirs = [os.path.abspath(d) for d in sys.argv[2:]]
    if len(run_dirs) < 2:
        raise SystemExit("need at least two run dirs to merge")

    stacks = [load_train_run_no_convert(_find_ckpt(d)) for d in run_dirs]

    merged = {}
    for key in ("best_params", "final_params"):
        if all(key in s for s in stacks):
            merged[key] = jax.tree_util.tree_map(
                lambda *xs: np.concatenate([np.asarray(x) for x in xs], axis=0),
                *[s[key] for s in stacks],
            )
    if not merged:
        raise SystemExit("no common params key (best_params/final_params) across inputs")

    seeds = None
    for key, tree in merged.items():
        leaf = jax.tree_util.tree_leaves(tree)[0]
        seeds = leaf.shape[0]
        print(f"merged {key}: {seeds} seeds, first leaf shape {leaf.shape}")

    os.makedirs(out_dir, exist_ok=True)
    ckpt_out = os.path.join(out_dir, "saved_train_run")
    if os.path.exists(ckpt_out):
        raise SystemExit(f"refusing to overwrite existing checkpoint {ckpt_out}")
    orbax.checkpoint.PyTreeCheckpointer().save(ckpt_out, merged)

    src_cfg = os.path.join(run_dirs[0], ".hydra")
    if os.path.isdir(src_cfg):
        shutil.copytree(src_cfg, os.path.join(out_dir, ".hydra"), dirs_exist_ok=True)

    print(f"wrote {seeds}-seed checkpoint: {ckpt_out}")
    print(f"sources: {len(run_dirs)} runs")


if __name__ == "__main__":
    main()
