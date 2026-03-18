#!/usr/bin/env bash
# Forced coord test: 8M steps, 5.6M warmup, dual critic, no JSD GAE.
# 4 configs: β=0/0.5 × ent=0.01/0.4. 1 seed each, all parallel.
#
# Usage:
#   ./run_forced_coord_test.sh <gpu>
# Example:
#   ./run_forced_coord_test.sh 6

GPU="${1:?Usage: ./run_forced_coord_test.sh <gpu>}"

TOTAL_TIMESTEPS=8e6
WARMUP_STEPS=5600000

mkdir -p logs

# β=0, default entropy
nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/forced_coord \
    algorithm.NUM_SEEDS=1 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0 \
    algorithm.USE_DUAL_CRITIC=true \
    label="dual_fc_b0_8M" \
    > logs/fc_b0_8M.log 2>&1 &

# β=0.5, default entropy
nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/forced_coord \
    algorithm.NUM_SEEDS=1 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.5 \
    algorithm.USE_DUAL_CRITIC=true \
    label="dual_fc_b05_8M" \
    > logs/fc_b05_8M.log 2>&1 &

# β=0, high entropy
nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/forced_coord \
    algorithm.NUM_SEEDS=1 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=0.40 \
    label="dual_fc_b0_ent040_8M" \
    > logs/fc_b0_ent040_8M.log 2>&1 &

# β=0.5, high entropy
nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/forced_coord \
    algorithm.NUM_SEEDS=1 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.5 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=0.40 \
    label="dual_fc_b05_ent040_8M" \
    > logs/fc_b05_ent040_8M.log 2>&1 &

echo "Launched 4 runs on GPU $GPU"
