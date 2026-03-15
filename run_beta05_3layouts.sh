#!/usr/bin/env bash
# Run JA-IPPO with beta=0.5 across 3 layouts, 5 seeds each.
# Each seed is a separate process (NUM_SEEDS=1) for isolation.
# All 5 seeds for a layout run concurrently on the same GPU,
# then the next layout starts.
#
# Usage:
#   ./run_beta05_3layouts.sh <gpu_id>

set -e

GPU="${1:?Usage: ./run_beta05_3layouts.sh <gpu_id>}"

LAYOUTS="cramped_room coord_ring forced_coord"
BETA=0.5
TOTAL_TIMESTEPS=5e6
WARMUP_STEPS=3500000

for LAYOUT in $LAYOUTS; do
    echo ""
    echo "=== $LAYOUT  beta=$BETA  launching 5 seeds in parallel ==="
    pids=()
    for SEED in 0 1 2 3 4; do
        echo "  Starting seed $SEED..."
        ./run_gpu.sh "$GPU" marl.run \
            -cn base_config_ja_ippo \
            task="overcooked-v1/$LAYOUT" \
            algorithm.NUM_SEEDS=1 \
            algorithm.TRAIN_SEED="$SEED" \
            algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
            algorithm.JA_WARMUP_ENV_STEPS="$WARMUP_STEPS" \
            algorithm.JA_BETA_MAX="$BETA" \
            label="beta05_seed${SEED}" \
            > "logs/${LAYOUT}_seed${SEED}.log" 2>&1 &
        pids+=($!)
    done

    echo "  Waiting for 5 seeds to finish..."
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: process $pid failed"
            failed=$((failed + 1))
        fi
    done

    if [ "$failed" -gt 0 ]; then
        echo "  $failed seed(s) failed for $LAYOUT — check logs/"
    else
        echo "  $LAYOUT complete"
    fi
done

echo ""
echo "=== All layouts done ==="
