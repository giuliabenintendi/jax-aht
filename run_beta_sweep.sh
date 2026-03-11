#!/usr/bin/env bash
# Beta sweep for JA-IPPO across Overcooked layouts.
# 5 betas × 5 seeds per run. Runs sequentially on a single GPU.
#
# Usage:
#   ./run_beta_sweep.sh <gpu_id> <layout>
#
# Layouts: cramped_room, forced_coord, coord_ring
#
# Examples:
#   ./run_beta_sweep.sh 6 cramped_room
#   ./run_beta_sweep.sh 6 coord_ring

set -e

GPU="${1:?Usage: ./run_beta_sweep.sh <gpu_id> <layout>}"
LAYOUT="${2:?Usage: ./run_beta_sweep.sh <gpu_id> <layout>}"

case "$LAYOUT" in
    cramped_room|forced_coord|coord_ring) ;;
    *)
        echo "Unknown layout: $LAYOUT"
        echo "Available: cramped_room, forced_coord, coord_ring"
        exit 1 ;;
esac

BETAS="0 0.1 0.25 0.5 1"
NUM_SEEDS=5
TOTAL_TIMESTEPS=5e6
WARMUP_STEPS=3500000

total=5
count=0

for BETA in $BETAS; do
    count=$((count + 1))
    echo ""
    echo "=== [$count/$total] $LAYOUT  beta=$BETA  seeds=$NUM_SEEDS ==="
    ./run_gpu.sh "$GPU" marl.run \
        -cn base_config_ja_ippo \
        task="overcooked-v1/$LAYOUT" \
        algorithm.NUM_SEEDS="$NUM_SEEDS" \
        algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
        algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
        algorithm.JA_BETA_MAX="$BETA" \
        label=beta_sweep
done

echo ""
echo "=== Beta sweep for $LAYOUT complete: $count/$total runs ==="
