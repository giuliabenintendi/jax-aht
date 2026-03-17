"""Merge multiple single-seed checkpoints into one multi-seed checkpoint.

Stacks final_params from separate runs so run_xp_seeds.py can consume them.

Usage:
    uv run python -m evaluation.merge_seeds \
        --checkpoints path1/saved_train_run path2/saved_train_run path3/saved_train_run \
        --output results/merged/saved_train_run
"""
import argparse
import os

import jax
import jax.numpy as jnp

from common.save_load_utils import load_train_run, save_train_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to single-seed saved_train_run directories")
    parser.add_argument("--output", required=True,
                        help="Output path for merged checkpoint")
    args = parser.parse_args()

    all_params = []
    for path in args.checkpoints:
        data = load_train_run(path)
        params = data["final_params"]
        num_seeds = jax.tree.leaves(params)[0].shape[0]
        if num_seeds == 1:
            all_params.append(params)
        else:
            # Multi-seed checkpoint: extract each seed individually
            for i in range(num_seeds):
                seed_params = jax.tree.map(lambda x: x[i:i+1], params)
                all_params.append(seed_params)
        print(f"Loaded {path}: {num_seeds} seed(s)")

    # Stack all params along seed axis
    merged_params = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *all_params)
    total_seeds = jax.tree.leaves(merged_params)[0].shape[0]
    print(f"Merged: {total_seeds} seeds total")

    # Save — copy hydra config from first checkpoint
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Copy .hydra config from first checkpoint
    first_run_dir = os.path.dirname(args.checkpoints[0])
    hydra_src = os.path.join(first_run_dir, ".hydra")
    hydra_dst = os.path.join(output_dir, ".hydra")
    if os.path.exists(hydra_src) and not os.path.exists(hydra_dst):
        import shutil
        shutil.copytree(hydra_src, hydra_dst)
        print(f"Copied .hydra config from {hydra_src}")

    merged_data = {
        "final_params": merged_params,
        "metrics": {},
        "checkpoints": {},
        "final_ckpt_idx": 0,
    }
    save_train_run(merged_data, output_dir, savename="saved_train_run")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
