#!/usr/bin/env bash
# Overnight: cramped_room β=0, dual critic, 3 configs. 1 seed each.
#
# Usage:
#   ./run_overnight.sh <gpu>
# Example:
#   ./run_overnight.sh 5

GPU="${1:?Usage: ./run_overnight.sh <gpu>}"

TOTAL_TIMESTEPS=5e6
WARMUP_STEPS=3500000

mkdir -p logs

# 1. Dual critic, no JSD GAE, high entropy
nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.NUM_SEEDS=1 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=0.45 \
    label="dual_cr_b0_nojsdgae_ent045" \
    > logs/dual_cr_b0_nojsdgae_ent045.log 2>&1 &

# 2. Dual critic, no JSD GAE, default entropy
nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.NUM_SEEDS=1 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0 \
    algorithm.USE_DUAL_CRITIC=true \
    label="dual_cr_b0_nojsdgae" \
    > logs/dual_cr_b0_nojsdgae.log 2>&1 &

# 3. Dual critic, JSD GAE on, high entropy
nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.NUM_SEEDS=1 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.DUAL_CRITIC_ACTOR_JA=true \
    algorithm.ENT_COEF=0.45 \
    label="dual_cr_b0_jsdgae_ent045" \
    > logs/dual_cr_b0_jsdgae_ent045.log 2>&1 &

echo "Launched 3 runs on GPU $GPU"
