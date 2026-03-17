#!/usr/bin/env bash
# Dual-critic JA-IPPO: coord_ring, beta=0.01, 5M, 3 seeds.
# Two sequential runs on the same GPU: JSD GAE off, then JSD GAE on.
#
# Usage:
#   ./run_overnight.sh <gpu>
# Example:
#   ./run_overnight.sh 6

set -e

GPU="${1:?Usage: ./run_overnight.sh <gpu>}"

TOTAL_TIMESTEPS=5e6
WARMUP_STEPS=3500000
BETA=0.01

mkdir -p logs

# Run 1: JSD GAE off (task-only policy gradient)
./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/coord_ring \
    algorithm.NUM_SEEDS=3 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX="$BETA" \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.DUAL_CRITIC_ACTOR_JA=false \
    label="dual_cring_b001_jsdgae_off" \
    > "logs/coord_ring_dual_jsdgae_off.log" 2>&1

# Run 2: JSD GAE on (task + JA policy gradient)
./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/coord_ring \
    algorithm.NUM_SEEDS=3 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
    algorithm.JA_BETA_MAX="$BETA" \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.DUAL_CRITIC_ACTOR_JA=true \
    label="dual_cring_b001_jsdgae_on" \
    > "logs/coord_ring_dual_jsdgae_on.log" 2>&1

echo "[$(date +%H:%M)] All runs complete"
