#!/usr/bin/env bash
# Dual-critic + high entropy + JSD GAE ON, 1 seed each, 3 layouts in parallel.
#
# Usage:
#   ./run_high_ent_jsdgae.sh <gpu>
# Example:
#   ./run_high_ent_jsdgae.sh 7

set -e

GPU="${1:?Usage: ./run_high_ent_jsdgae.sh <gpu>}"

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
    algorithm.JA_BETA_MAX=0.25 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.DUAL_CRITIC_ACTOR_JA=true \
    algorithm.ENT_COEF=0.45 \
    label="dual_cr_b025_jsdgae_ent045" \
    > logs/cr_dual_jsdgae_ent045.log 2>&1 &

nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/coord_ring \
    algorithm.NUM_SEEDS=1 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.001 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.DUAL_CRITIC_ACTOR_JA=true \
    algorithm.ENT_COEF=0.45 \
    label="dual_cring_b0001_jsdgae_ent045" \
    > logs/cring_dual_jsdgae_ent045.log 2>&1 &

nohup ./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/forced_coord \
    algorithm.NUM_SEEDS=1 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX=0.5 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.DUAL_CRITIC_ACTOR_JA=true \
    algorithm.ENT_COEF=0.40 \
    label="dual_fc_b05_jsdgae_ent040" \
    > logs/fc_dual_jsdgae_ent040.log 2>&1 &

echo "Launched 3 runs on GPU $GPU"
