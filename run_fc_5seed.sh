#!/usr/bin/env bash
# Forced coord 8M, 5 seeds, dual critic, no JSD GAE.
# β=0 (ent=0.01) and β=0.5 (ent=0.4) in parallel.
#
# Usage:
#   ./run_fc_5seed.sh <gpu>
# Example:
#   ./run_fc_5seed.sh 6

GPU="${1:?Usage: ./run_fc_5seed.sh <gpu>}"

TOTAL_TIMESTEPS=8e6
WARMUP_STEPS=5600000

mkdir -p logs

nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/forced_coord \
    algorithm.NUM_SEEDS=5 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0 \
    algorithm.USE_DUAL_CRITIC=true \
    label="dual_fc_b0_8M_5s" \
    > logs/fc_b0_8M_5s.log 2>&1 &

nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/forced_coord \
    algorithm.NUM_SEEDS=5 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.5 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=0.40 \
    label="dual_fc_b05_ent040_8M_5s" \
    > logs/fc_b05_ent040_8M_5s.log 2>&1 &

echo "Launched 2 runs on GPU $GPU"
