"""Diagnose the JA+shaping snapshot-vs-best gap.

Reads chunk_scores.json for the likely-thunder-1466 run and prints:
  - distribution of best-chunk return per seed (mean / median / min / max)
  - all top-level keys (so we can spot a per-chunk-per-seed array if present)
  - if a per-chunk-per-seed return array is found, prints chunk-76 mean across all 48 seeds

Usage:
    uv run python evaluation/card_game/diag_ja_shaping_gap.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCORES = Path(
    "/scratch/benintendi/jax-aht/checkpoints/"
    "likely-thunder-1466_card-game-op-delib-actions_ja_ippo_op_ja_shaped_48s_15M_s48_21052026/"
    "chunk_scores.json"
)


def main() -> None:
    s = json.loads(SCORES.read_text())

    br = np.array(s["best_chunk_return_per_seed"])
    print(f"best-chunk return per seed (training-time SP under OP):")
    print(f"  48-seed mean   = {br.mean():.3f}")
    print(f"  48-seed median = {np.median(br):.3f}")
    print(f"  min / max      = {br.min():.3f} / {br.max():.3f}")
    print()

    print("all keys in chunk_scores.json:")
    for k in s.keys():
        v = s[k]
        if isinstance(v, list):
            arr = np.asarray(v)
            print(f"  {k:40s}  list shape={arr.shape}  dtype={arr.dtype}")
        elif isinstance(v, (int, float, str, bool)):
            print(f"  {k:40s}  {v}")
        else:
            print(f"  {k:40s}  {type(v).__name__}")
    print()

    # If there's a per-chunk-per-seed array, compute the 48-seed mean at the final chunk
    # to see whether the gap to 0.72 is snapshot instability (low) or subset bias (high).
    for k, v in s.items():
        if not isinstance(v, list):
            continue
        arr = np.asarray(v)
        if arr.ndim == 2 and "return" in k.lower():
            print(f"per-chunk-per-seed table: {k}  shape={arr.shape}")
            # convention: [chunks, seeds] or [seeds, chunks] — try both
            if arr.shape[1] == 48:
                last_chunk = arr[-1]
                print(f"  shape[0]={arr.shape[0]} (chunks?), shape[1]=48 (seeds?)")
                print(f"  chunk[-1] across 48 seeds: mean={last_chunk.mean():.3f} "
                      f"std={last_chunk.std(ddof=1):.3f}")
            elif arr.shape[0] == 48:
                last_chunk = arr[:, -1]
                print(f"  shape[0]=48 (seeds?), shape[1]={arr.shape[1]} (chunks?)")
                print(f"  chunk[-1] across 48 seeds: mean={last_chunk.mean():.3f} "
                      f"std={last_chunk.std(ddof=1):.3f}")

    # Also check best_chunk_idx_per_seed distribution
    if "best_chunk_idx_per_seed" in s:
        idx = np.array(s["best_chunk_idx_per_seed"])
        num_ckpts = int(s.get("num_ckpts", idx.max() + 1))
        print()
        print(f"best_chunk_idx_per_seed (num_ckpts={num_ckpts}):")
        print(f"  mean idx   = {idx.mean():.1f}  (out of {num_ckpts-1})")
        print(f"  median idx = {int(np.median(idx))}")
        print(f"  min / max  = {idx.min()} / {idx.max()}")
        # Histogram in quintiles of training
        bins = np.linspace(0, num_ckpts - 1, 6)
        hist, _ = np.histogram(idx, bins=bins)
        print(f"  seeds whose best chunk is in 1st/2nd/3rd/4th/5th fifth of training:")
        print(f"    {hist.tolist()}")


if __name__ == "__main__":
    main()
