#!/usr/bin/env bash
# XP evaluation for all 6 LBF runs. Split across 2 GPUs.
#
# Usage:
#   ./run_lbf_xp.sh <gpu_a> <gpu_b>
# Example:
#   ./run_lbf_xp.sh 5 6

set -e

GPU_A="${1:?Usage: ./run_lbf_xp.sh <gpu_a> <gpu_b>}"
GPU_B="${2:?Usage: ./run_lbf_xp.sh <gpu_a> <gpu_b>}"

mkdir -p logs

# GPU_A: 3-food (sequential)
(
    echo "[$(date +%H:%M)] 3-food beta=0"
    ./run_gpu.sh "$GPU_A" evaluation.run_xp_seeds \
        --checkpoint results/lbf-image/ja_ippo/lbf3_b0/2026-03-17_11-21-35/saved_train_run
    echo "[$(date +%H:%M)] 3-food beta=0.001"
    ./run_gpu.sh "$GPU_A" evaluation.run_xp_seeds \
        --checkpoint results/lbf-image/ja_ippo/lbf3_b0001/2026-03-18_08-43-49/saved_train_run
    echo "[$(date +%H:%M)] 3-food beta=0.002"
    ./run_gpu.sh "$GPU_A" evaluation.run_xp_seeds \
        --checkpoint results/lbf-image/ja_ippo/lbf3_b0002/2026-03-18_08-43-49/saved_train_run
) > logs/lbf3_xp.log 2>&1 &

# GPU_B: 10-food (sequential)
(
    echo "[$(date +%H:%M)] 10-food beta=0"
    ./run_gpu.sh "$GPU_B" evaluation.run_xp_seeds \
        --checkpoint results/lbf-image-10food/ja_ippo/lbf10_b0/2026-03-17_11-21-35/saved_train_run
    echo "[$(date +%H:%M)] 10-food beta=0.001"
    ./run_gpu.sh "$GPU_B" evaluation.run_xp_seeds \
        --checkpoint results/lbf-image-10food/ja_ippo/lbf10_b0001/2026-03-18_08-43-49/saved_train_run
    echo "[$(date +%H:%M)] 10-food beta=0.002"
    ./run_gpu.sh "$GPU_B" evaluation.run_xp_seeds \
        --checkpoint results/lbf-image-10food/ja_ippo/lbf10_b0002/2026-03-18_08-43-49/saved_train_run
) > logs/lbf10_xp.log 2>&1 &

wait
echo "[$(date +%H:%M)] All LBF XP evals complete"
