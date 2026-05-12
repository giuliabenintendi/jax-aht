"""Read the raw orbax checkpoint and print scalar_embed kernel shape."""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import orbax.checkpoint as ocp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True,
                   help="Path to saved_train_run/ directory")
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    print(f"ckpt_dir = {ckpt_dir}")
    print(f"contents: {sorted(p.name for p in ckpt_dir.iterdir())}\n")

    mgr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions())
    steps = mgr.all_steps()
    print(f"steps: {steps}\n")

    latest = steps[-1] if steps else None
    if latest is None:
        print("no steps found")
        return

    restored = mgr.restore(latest)

    def walk(tree, prefix=""):
        if hasattr(tree, "keys"):
            for k, v in tree.items():
                walk(v, f"{prefix}/{k}")
        else:
            shape = getattr(tree, "shape", None)
            if shape is not None and "scalar_embed" in prefix:
                print(f"  {prefix}: shape={shape}")

    walk(restored)


if __name__ == "__main__":
    main()
