#!/usr/bin/env bash
# Coord ring: low beta sweep (0.01, 0.001), 5M, 3 seeds concurrent.
# Runs both betas in parallel on two GPUs.
#
# Usage: ./run_coord_ring_low_beta.sh <gpu_a> <gpu_b>
# Example: ./run_coord_ring_low_beta.sh 5 6

set -e

GPU_A="${1:?Usage: ./run_coord_ring_low_beta.sh <gpu_a> <gpu_b>}"
GPU_B="${2:?Usage: ./run_coord_ring_low_beta.sh <gpu_a> <gpu_b>}"

TOTAL_TIMESTEPS=5e6
WARMUP_STEPS=700000

mkdir -p logs

echo "[$(date +%H:%M)] Starting coord_ring low beta sweep"
echo "  GPU $GPU_A: beta=0.01 (3 seeds)"
echo "  GPU $GPU_B: beta=0.001 (3 seeds)"

# GPU_A: beta=0.01
for SEED in 0 1 2; do
    ./run_gpu.sh "$GPU_A" marl.run \
        -cn base_config_ja_ippo \
        task=overcooked-v1/coord_ring \
        algorithm.NUM_SEEDS=1 \
        algorithm.TRAIN_SEED="$SEED" \
        algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
        algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
        algorithm.JA_BETA_MAX=0.01 \
        label="low_beta_b001_s${SEED}" \
        > "logs/coord_ring_b001_seed${SEED}.log" 2>&1 &
done

# GPU_B: beta=0.001
for SEED in 0 1 2; do
    ./run_gpu.sh "$GPU_B" marl.run \
        -cn base_config_ja_ippo \
        task=overcooked-v1/coord_ring \
        algorithm.NUM_SEEDS=1 \
        algorithm.TRAIN_SEED="$SEED" \
        algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
        algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
        algorithm.JA_BETA_MAX=0.001 \
        label="low_beta_b0001_s${SEED}" \
        > "logs/coord_ring_b0001_seed${SEED}.log" 2>&1 &
done

wait
echo "[$(date +%H:%M)] All done"
