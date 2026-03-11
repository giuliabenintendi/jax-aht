#!/usr/bin/env bash
# Run XP evaluation on all beta_sweep checkpoints for a layout.
# Finds all saved_train_run dirs under results/<layout>/ja_ippo/beta_sweep/
# and runs cross-play evaluation on each.
#
# Usage: ./run_beta_sweep_eval.sh <gpu_id> <layout>
#
# Examples:
#   ./run_beta_sweep_eval.sh 6 cramped_room
#   ./run_beta_sweep_eval.sh 6 coord_ring

set -e

GPU="${1:?Usage: ./run_beta_sweep_eval.sh <gpu_id> <layout>}"
LAYOUT="${2:?Usage: ./run_beta_sweep_eval.sh <gpu_id> <layout>}"

case "$LAYOUT" in
    cramped_room|coord_ring|forced_coord)
        EVAL_TASK_CFG="overcooked-v1-image/$LAYOUT"
        RESULT_DIR="results/overcooked-v1/$LAYOUT/ja_ippo/beta_sweep" ;;
    *)
        echo "Unknown layout: $LAYOUT"
        echo "Available: cramped_room, forced_coord, coord_ring"
        exit 1 ;;
esac

CKPTS=$(ls -td $RESULT_DIR/*/saved_train_run 2>/dev/null)
if [ -z "$CKPTS" ]; then
    echo "ERROR: no checkpoints found at $RESULT_DIR/*/saved_train_run"
    exit 1
fi

total=$(echo "$CKPTS" | wc -l | tr -d ' ')
count=0

echo "=== Beta sweep eval: $LAYOUT ($total checkpoints) ==="

for CKPT_PATH in $CKPTS; do
    count=$((count + 1))
    RUN_DIR=$(dirname "$CKPT_PATH")
    RUN_NAME=$(basename "$RUN_DIR")
    echo ""
    echo "=== [$count/$total] $RUN_NAME ==="
    echo "    checkpoint: $CKPT_PATH"

    CUDA_VISIBLE_DEVICES="$GPU" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    LD_LIBRARY_PATH="" \
    uv run python -m evaluation.run_xp_seeds \
        --task "$EVAL_TASK_CFG" \
        --checkpoint "$CKPT_PATH"
done

echo ""
echo "=== Beta sweep eval complete: $count/$total ==="
