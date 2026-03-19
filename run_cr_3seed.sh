#!/usr/bin/env bash
# Cramped room, 3 seeds, dual critic, no JSD GAE, ent=0.45.
# β=0 on GPU_A, β=0.25 on GPU_B.
#
# Usage:
#   ./run_cr_3seed.sh <gpu_a> <gpu_b>
# Example:
#   ./run_cr_3seed.sh 1 5

GPU_A="${1:?Usage: ./run_cr_3seed.sh <gpu_a> <gpu_b>}"
GPU_B="${2:?Usage: ./run_cr_3seed.sh <gpu_a> <gpu_b>}"

TOTAL_TIMESTEPS=5e6
WARMUP_STEPS=3500000

mkdir -p logs

nohup ./run_gpu.sh "$GPU_A" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.NUM_SEEDS=3 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=0.45 \
    label="dual_cr_b0_ent045_3s" \
    > logs/dual_cr_b0_ent045_3s.log 2>&1 &

nohup ./run_gpu.sh "$GPU_B" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.NUM_SEEDS=3 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.25 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=0.45 \
    label="dual_cr_b025_ent045_3s" \
    > logs/dual_cr_b025_ent045_3s.log 2>&1 &

echo "Launched 2 runs on GPU $GPU_A and $GPU_B"
