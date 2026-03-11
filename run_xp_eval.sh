#!/usr/bin/env bash
# Run within-layout cross-play evaluation on JA-IPPO trained checkpoints.
# Usage: ./run_xp_eval.sh <gpu_id> <task>
#
# Tasks: cramped_room, coord_ring, forced_coord, lbf-image-10food
# Finds the latest xp_seeds training run for the given task.
#
# Examples:
#   ./run_xp_eval.sh 1 cramped_room
#   ./run_xp_eval.sh 5 lbf-image-10food

set -e

GPU="${1:?Usage: ./run_xp_eval.sh <gpu_id> <task>}"
TASK="${2:?Usage: ./run_xp_eval.sh <gpu_id> <task>}"

# Map short task names to eval task config and training result path
case "$TASK" in
    cramped_room|coord_ring|forced_coord)
        EVAL_TASK_CFG="overcooked-v1-image/$TASK"
        RESULT_DIR="results/overcooked-v1/$TASK/ja_ippo/xp_seeds" ;;
    lbf-image-10food)
        EVAL_TASK_CFG="lbf-image-10food"
        RESULT_DIR="results/lbf-image-10food/ja_ippo/xp_seeds" ;;
    *)
        echo "Unknown task: $TASK"
        echo "Available: cramped_room, coord_ring, forced_coord, lbf-image-10food"
        exit 1 ;;
esac

CKPT_PATH=$(ls -td $RESULT_DIR/*/saved_train_run 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: no checkpoint found at $RESULT_DIR/*/saved_train_run"
    exit 1
fi

echo "=== XP eval: $TASK ==="
echo "    checkpoint: $CKPT_PATH"

CUDA_VISIBLE_DEVICES="$GPU" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
LD_LIBRARY_PATH="" \
uv run python -m evaluation.run \
    --config-name=xp_ja_seeds \
    task=$EVAL_TASK_CFG \
    checkpoint_path="$CKPT_PATH"
