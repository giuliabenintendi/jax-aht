#!/usr/bin/env bash
# LBF beta sweep: 3-food and 10-food, beta=0.001 and 0.002, 2M, 3 seeds.
# Split across 2 GPUs. Beta=0 already ran.
#
# Usage:
#   ./run_lbf_beta.sh <gpu_a> <gpu_b>
# Example:
#   ./run_lbf_beta.sh 5 6

set -e

GPU_A="${1:?Usage: ./run_lbf_beta.sh <gpu_a> <gpu_b>}"
GPU_B="${2:?Usage: ./run_lbf_beta.sh <gpu_a> <gpu_b>}"

TOTAL_TIMESTEPS=2e6
WARMUP_STEPS=1400000

mkdir -p logs

# GPU_A: 3-food beta=0.001 + 10-food beta=0.001
nohup ./run_gpu.sh "$GPU_A" marl.run \
    -cn base_config_ja_ippo \
    task=lbf-image \
    algorithm.NUM_SEEDS=3 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.001 \
    label="lbf3_b0001" \
    > "logs/lbf3_b0001.log" 2>&1 &

nohup ./run_gpu.sh "$GPU_A" marl.run \
    -cn base_config_ja_ippo \
    task=lbf-image-10food \
    algorithm.NUM_SEEDS=3 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.001 \
    label="lbf10_b0001" \
    > "logs/lbf10_b0001.log" 2>&1 &

# GPU_B: 3-food beta=0.002 + 10-food beta=0.002
nohup ./run_gpu.sh "$GPU_B" marl.run \
    -cn base_config_ja_ippo \
    task=lbf-image \
    algorithm.NUM_SEEDS=3 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.002 \
    label="lbf3_b0002" \
    > "logs/lbf3_b0002.log" 2>&1 &

nohup ./run_gpu.sh "$GPU_B" marl.run \
    -cn base_config_ja_ippo \
    task=lbf-image-10food \
    algorithm.NUM_SEEDS=3 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.002 \
    label="lbf10_b0002" \
    > "logs/lbf10_b0002.log" 2>&1 &

echo "Launched 4 runs on GPU $GPU_A and $GPU_B"
