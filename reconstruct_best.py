"""
Reconstruct best_params from per-chunk folders + chunk_scores.json after a post-train
crash. Writes a saved_train_run/ that `run_xp_seeds.py --checkpoint --use-best` accepts.

Pure numpy + orbax (no jax import) so the saved checkpoint is device-agnostic --
can be loaded on GPU even though this script runs on CPU.

Run from the repo root:
    uv run reconstruct_best.py \\
        --ckpt-root /scratch/benintendi/jax-aht/checkpoints/likely-thunder-1466_card-game-op-delib-actions_ja_ippo_op_ja_shaped_48s_15M_s48_21052026 \\
        --hydra-dir /scratch/benintendi/jax-aht/results/card-game-op-delib-actions/ja_ippo/op_ja_shaped_48s/2026-05-21_23-14-01
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import numpy as np
import orbax.checkpoint as ocp
from flax.training import orbax_utils


def to_numpy_tree(obj):
    """Recursively convert any array-like leaves to numpy. Strips jax sharding info."""
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
    restored = checkpointer.restore(path)
    return to_numpy_tree(restored)


def gather_seed(s: int, tree):
    if isinstance(tree, dict):
        return {k: gather_seed(s, v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(gather_seed(s, x) for x in tree)
    return tree[s]


def stack_trees(trees):
    sample = trees[0]
    if isinstance(sample, dict):
        return {k: stack_trees([t[k] for t in trees]) for k in sample}
    if isinstance(sample, (list, tuple)):
        return type(sample)(stack_trees([t[i] for t in trees]) for i in range(len(sample)))
    return np.stack(trees)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True,
                    help="folder containing ckpt_NN_* subdirs and chunk_scores.json")
    ap.add_argument("--hydra-dir", required=True,
                    help="original hydra output dir (must contain .hydra/config.yaml). "
                         "saved_train_run/ is written here so XP eval picks up the right config.")
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_root)
    hydra_dir = Path(args.hydra_dir)

    scores_path = ckpt_root / "chunk_scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"no chunk_scores.json at {scores_path}")
    if not (hydra_dir / ".hydra" / "config.yaml").exists():
        raise FileNotFoundError(f"no .hydra/config.yaml in {hydra_dir}")

    scores = json.loads(scores_path.read_text())
    best_idx = scores["best_ckpt_idx_per_seed"]
    ckpt_paths = scores["ckpt_folder_paths"]
    num_seeds = scores["num_seeds"]
    num_ckpts = scores["num_ckpts"]
    if len(best_idx) != num_seeds:
        raise AssertionError(
            f"best_ckpt_idx_per_seed has {len(best_idx)} entries but num_seeds={num_seeds}"
        )

    unique = sorted(set(best_idx))
    print(f"reconstructing best_params: {num_seeds} seeds, {len(unique)} unique chunks "
          f"(out of {num_ckpts})")

    cache = {}
    for ci in unique:
        path = ckpt_paths[ci]
        print(f"  loading chunk {ci}: {path}")
        cache[ci] = load_chunk_numpy(path)

    print("stacking per-seed best params...")
    per_seed = [gather_seed(s, cache[best_idx[s]]) for s in range(num_seeds)]
    best_params = stack_trees(per_seed)

    fci = num_ckpts - 1
    print(f"loading final chunk ({fci}) as final_params...")
    final_params = cache[fci] if fci in cache else load_chunk_numpy(ckpt_paths[fci])

    out = {
        "best_params": best_params,
        "final_params": final_params,
        "ckpt_env_steps": np.asarray(scores["ckpt_env_steps"], dtype=np.int64),
        "best_ckpt_idx": np.asarray(best_idx, dtype=np.int32),
    }
    out = to_numpy_tree(out)

    save_path = str(hydra_dir / "saved_train_run")
    if os.path.exists(save_path):
        print(f"removing existing {save_path}...")
        shutil.rmtree(save_path)

    print(f"writing saved_train_run to {save_path}...")
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(out)
    checkpointer.save(save_path, out, save_args=save_args)

    print()
    print(f"DONE. saved_train_run written to: {save_path}")
    print()
    print("Launch standalone XP on GPU 4:")
    print(f"  ./run_gpu.sh 4 evaluation.run_xp_seeds \\")
    print(f"      --checkpoint {save_path} \\")
    print(f"      --use-best --no-xp-videos")


if __name__ == "__main__":
    main()
