#!/usr/bin/env bash
# Overnight: dual-critic JA-IPPO, 5M timesteps, 3 seeds concurrent per layout.
#
# GPU_A: cramped_room (beta=0.25) + forced_coord (beta=0.5) — 6 seeds sequential
# GPU_B: coord_ring (beta=0.001) — 3 seeds
#
# Usage:
#   ./run_overnight.sh <gpu_a> <gpu_b>
# Example:
#   ./run_overnight.sh 5 6

set -e

GPU_A="${1:?Usage: ./run_overnight.sh <gpu_a> <gpu_b>}"
GPU_B="${2:?Usage: ./run_overnight.sh <gpu_a> <gpu_b>}"

TOTAL_TIMESTEPS=5e6
WARMUP_STEPS=3500000

mkdir -p logs

# GPU_A: cramped_room (beta=0.25), then forced_coord (beta=0.5)
(
    ./run_gpu.sh "$GPU_A" marl.run \
        -cn base_config_ja_ippo \
        task=overcooked-v1/cramped_room \
        algorithm.NUM_SEEDS=3 \
        algorithm.TRAIN_SEED=42 \
        algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
        algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
        algorithm.JA_BETA_MAX=0.25 \
        algorithm.USE_DUAL_CRITIC=true \
        label="dual_cr_b025" \
        > "logs/cramped_room_dual.log" 2>&1

    ./run_gpu.sh "$GPU_A" marl.run \
        -cn base_config_ja_ippo \
        task=overcooked-v1/forced_coord \
        algorithm.NUM_SEEDS=3 \
        algorithm.TRAIN_SEED=42 \
        algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
        algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
        algorithm.JA_BETA_MAX=0.5 \
        algorithm.USE_DUAL_CRITIC=true \
        label="dual_fc_b05" \
        > "logs/forced_coord_dual.log" 2>&1
) &

# GPU_B: coord_ring (beta=0.001)
./run_gpu.sh "$GPU_B" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/coord_ring \
    algorithm.NUM_SEEDS=3 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.001 \
    algorithm.USE_DUAL_CRITIC=true \
    label="dual_cring_b0001" \
    > "logs/coord_ring_dual.log" 2>&1 &

wait
echo "[$(date +%H:%M)] All runs complete"
