#!/usr/bin/env bash
# Run XP evaluation on all multi-seed checkpoints across all layouts.
# Single-seed checkpoints are automatically skipped by run_xp_seeds.py.
#
# Usage: ./run_all_xp.sh <gpu_id>

set -e

GPU="${1:?Usage: ./run_all_xp.sh <gpu_id>}"

BASE_DIR="results/overcooked-v1"

CKPTS=$(ls -td $BASE_DIR/*/ja_ippo/beta_sweep/*/saved_train_run 2>/dev/null)
if [ -z "$CKPTS" ]; then
    echo "ERROR: no checkpoints found under $BASE_DIR"
    exit 1
fi

total=$(echo "$CKPTS" | wc -l | tr -d ' ')
count=0

echo "=== XP eval: all layouts ($total checkpoints) ==="

for CKPT_PATH in $CKPTS; do
    count=$((count + 1))
    RUN_DIR=$(dirname "$CKPT_PATH")
    RUN_NAME=$(basename "$RUN_DIR")
    LAYOUT=$(echo "$CKPT_PATH" | sed "s|$BASE_DIR/||" | cut -d/ -f1)
    echo ""
    echo "=== [$count/$total] $LAYOUT / $RUN_NAME ==="
    echo "    checkpoint: $CKPT_PATH"

    CUDA_VISIBLE_DEVICES="$GPU" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    LD_LIBRARY_PATH="" \
    uv run python -m evaluation.run_xp_seeds \
        --checkpoint "$CKPT_PATH"
done

echo ""
echo "=== XP eval complete: $count/$total ==="
