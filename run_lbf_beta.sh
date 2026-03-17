#!/usr/bin/env bash
# LBF beta sweep: 3-food and 10-food, beta=0/0.001/0.002, 2M, 3 seeds.
# Single-critic JA-IPPO. Two tasks run in parallel on the same GPU.
#
# Usage:
#   ./run_lbf_beta.sh <gpu>
# Example:
#   ./run_lbf_beta.sh 6

set -e

GPU="${1:?Usage: ./run_lbf_beta.sh <gpu>}"

TOTAL_TIMESTEPS=2e6
WARMUP_STEPS=1400000

mkdir -p logs

# 3-food (lbf-image): beta=0, 0.001, 0.002 sequentially
(
    for BETA in 0 0.001 0.002; do
        LABEL="lbf3_b${BETA}"
        echo "[$(date +%H:%M)] Starting lbf-image beta=${BETA}"
        ./run_gpu.sh "$GPU" marl.run \
            -cn base_config_ja_ippo \
            task=lbf-image \
            algorithm.NUM_SEEDS=3 \
            algorithm.TRAIN_SEED=42 \
            algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
            algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
            algorithm.JA_BETA_MAX="$BETA" \
            label="$LABEL" \
            > "logs/lbf3_b${BETA}.log" 2>&1
    done
) &

# 10-food (lbf-image-10food): beta=0, 0.001, 0.002 sequentially
(
    for BETA in 0 0.001 0.002; do
        LABEL="lbf10_b${BETA}"
        echo "[$(date +%H:%M)] Starting lbf-image-10food beta=${BETA}"
        ./run_gpu.sh "$GPU" marl.run \
            -cn base_config_ja_ippo \
            task=lbf-image-10food \
            algorithm.NUM_SEEDS=3 \
            algorithm.TRAIN_SEED=42 \
            algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
            algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
            algorithm.JA_BETA_MAX="$BETA" \
            label="$LABEL" \
            > "logs/lbf10_b${BETA}.log" 2>&1
    done
) &

wait
echo "[$(date +%H:%M)] All LBF runs complete"
