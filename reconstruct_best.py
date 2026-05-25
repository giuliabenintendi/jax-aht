"""
Reconstruct best_params from per-chunk folders + chunk_scores.json after a post-train
crash. Writes a saved_train_run/ that `run_xp_seeds.py --checkpoint --use-best` accepts.

Run from the repo root.

Example:
    uv run reconstruct_best.py \\
        --ckpt-root /scratch/benintendi/jax-aht/checkpoints/likely-thunder-1466_card-game-op-delib-actions_ja_ippo_op_ja_shaped_48s_15M_s48_21052026 \\
        --hydra-dir /scratch/benintendi/jax-aht/results/card-game-op-delib-actions/ja_ippo/op_ja_shaped_48s/<TIMESTAMP>
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import jax
import numpy as np
from common.save_load_utils import load_train_run, save_train_run


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
    hydra_cfg = hydra_dir / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
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
        cache[ci] = load_train_run(path)

    def gather(s: int):
        c = cache[best_idx[s]]
        return jax.tree.map(lambda x: np.asarray(x)[s], c)

    print("stacking per-seed best params...")
    per_seed = [gather(s) for s in range(num_seeds)]
    best_params = jax.tree.map(lambda *xs: np.stack(xs), *per_seed)

    fci = num_ckpts - 1
    print(f"loading final chunk ({fci}) as final_params...")
    final_params = cache[fci] if fci in cache else load_train_run(ckpt_paths[fci])

    out = {
        "best_params": best_params,
        "final_params": final_params,
        "ckpt_env_steps": np.asarray(scores["ckpt_env_steps"], dtype=np.int64),
        "best_ckpt_idx": np.asarray(best_idx, dtype=np.int32),
    }

    save_path = save_train_run(out, str(hydra_dir), "saved_train_run")
    print()
    print(f"DONE. saved_train_run written to: {save_path}")
    print()
    print("Launch standalone XP on GPU 4:")
    print(f"  ./run_gpu.sh 4 evaluation.run_xp_seeds \\")
    print(f"      --checkpoint {save_path} \\")
    print(f"      --use-best --no-xp-videos")


if __name__ == "__main__":
    main()
