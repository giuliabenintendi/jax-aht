#!/usr/bin/env bash
# Sweep ENT_COEF on card-game-op (OP + comm) with 4 seeds per value.
# Usage: ./run_ent_sweep.sh [gpu_id]
set -e

GPU=${1:-7}

for ENT in 0.03 0.05 0.1; do
    echo "=== ENT_COEF=$ENT (GPU=$GPU, 4 seeds, 5M steps) ==="
    ./run_gpu.sh "$GPU" marl.run \
        task=card-game-op \
        algorithm=ja_ippo/card-game-op \
        algorithm.NUM_SEEDS=4 \
        algorithm.ENT_COEF="$ENT" \
        label="4s_5M_ent${ENT}"
done
