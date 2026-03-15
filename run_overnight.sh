#!/usr/bin/env bash
# Overnight: coord_ring on GPU_A, forced_coord on GPU_B.
# Beta=0.5, norm_off, 5M timesteps, 3 seeds concurrent per GPU.
#
# Usage:
#   ./run_overnight.sh <gpu_a> <gpu_b>
# Example:
#   ./run_overnight.sh 5 6

set -e

GPU_A="${1:?Usage: ./run_overnight.sh <gpu_a> <gpu_b>}"
GPU_B="${2:?Usage: ./run_overnight.sh <gpu_a> <gpu_b>}"

BETA=0.5
TOTAL_TIMESTEPS=5e6
WARMUP_STEPS=3500000

mkdir -p logs

# GPU_A: coord_ring, 3 seeds
for SEED in 0 1 2; do
    ./run_gpu.sh "$GPU_A" marl.run \
        -cn base_config_ja_ippo \
        task=overcooked-v1/coord_ring \
        algorithm.NUM_SEEDS=1 \
        algorithm.TRAIN_SEED="$SEED" \
        algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
        algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
        algorithm.JA_BETA_MAX="$BETA" \
        algorithm.NORMALIZE_REWARDS=false \
        label="norm_off_b05_s${SEED}" \
        > "logs/coord_ring_seed${SEED}.log" 2>&1 &
done

# GPU_B: forced_coord, 3 seeds
for SEED in 0 1 2; do
    ./run_gpu.sh "$GPU_B" marl.run \
        -cn base_config_ja_ippo \
        task=overcooked-v1/forced_coord \
        algorithm.NUM_SEEDS=1 \
        algorithm.TRAIN_SEED="$SEED" \
        algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
        algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
        algorithm.JA_BETA_MAX="$BETA" \
        algorithm.NORMALIZE_REWARDS=false \
        label="norm_off_b05_s${SEED}" \
        > "logs/forced_coord_seed${SEED}.log" 2>&1 &
done

wait
echo "[$(date +%H:%M)] All runs complete"
