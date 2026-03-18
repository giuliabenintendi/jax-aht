#!/usr/bin/env bash
# Single-critic + high entropy, cramped room, 1 seed each.
# Beta=0 and Beta=0.25 in parallel.
#
# Usage:
#   ./run_high_ent_single.sh <gpu>
# Example:
#   ./run_high_ent_single.sh 1

set -e

GPU="${1:?Usage: ./run_high_ent_single.sh <gpu>}"

TOTAL_TIMESTEPS=5e6
WARMUP_STEPS=3500000

mkdir -p logs

nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.NUM_SEEDS=1 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0 \
    algorithm.ENT_COEF=0.45 \
    label="cr_b0_ent045" \
    > logs/cr_b0_ent045.log 2>&1 &

nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.NUM_SEEDS=1 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.25 \
    algorithm.ENT_COEF=0.45 \
    label="cr_b025_ent045" \
    > logs/cr_b025_ent045.log 2>&1 &

echo "Launched 2 runs on GPU $GPU"
