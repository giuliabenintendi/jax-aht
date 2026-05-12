"""Read the raw orbax checkpoint and print scalar_embed kernel shape."""
from __future__ import annotations

import argparse
from pathlib import Path

from common.save_load_utils import load_train_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True,
                   help="Path to saved_train_run/ directory")
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    print(f"ckpt_dir = {ckpt_dir}")
    print(f"contents: {sorted(p.name for p in ckpt_dir.iterdir())}\n")

    restored = load_train_run(str(ckpt_dir))
    print(f"top-level type: {type(restored).__name__}")
    if isinstance(restored, (list, tuple)):
        print(f"  length: {len(restored)}")
        roots = list(enumerate(restored))
    elif hasattr(restored, "keys"):
        roots = list(restored.items())
    else:
        roots = [("root", restored)]

    def walk(tree, prefix=""):
        if hasattr(tree, "keys"):
            for k, v in tree.items():
                walk(v, f"{prefix}/{k}")
        else:
            shape = getattr(tree, "shape", None)
            if shape is None:
                return
            # print ALL leaves under actor_lstm / critic_lstm to see the param tree
            if "lstm" in prefix or "scalar_embed" in prefix:
                print(f"  {prefix}: shape={shape}")

    for key, sub in roots:
        print(f"\n--- root[{key}] ---")
        walk(sub)


if __name__ == "__main__":
    main()
