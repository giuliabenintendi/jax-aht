#!/usr/bin/env bash
# Run within-layout cross-play evaluation on JA-IPPO trained checkpoints.
# Finds the latest xp_seeds training run for each of the 5 seeds.
#
# Usage: ./run_xp_eval.sh <gpu_id> <task>
#
# Tasks: cramped_room, coord_ring, forced_coord, lbf-image-10food
#
# Examples:
#   ./run_xp_eval.sh 1 cramped_room
#   ./run_xp_eval.sh 5 lbf-image-10food

set -e

GPU="${1:?Usage: ./run_xp_eval.sh <gpu_id> <task>}"
TASK="${2:?Usage: ./run_xp_eval.sh <gpu_id> <task>}"

NUM_SEEDS=5

# Map short task names to eval task config and training result path
case "$TASK" in
    cramped_room|coord_ring|forced_coord)
        EVAL_TASK_CFG="overcooked-v1-image/$TASK"
        RESULT_BASE="results/overcooked-v1/$TASK/ja_ippo/xp_seeds" ;;
    lbf-image-10food)
        EVAL_TASK_CFG="lbf-image-10food"
        RESULT_BASE="results/lbf-image-10food/ja_ippo/xp_seeds" ;;
    *)
        echo "Unknown task: $TASK"
        echo "Available: cramped_room, coord_ring, forced_coord, lbf-image-10food"
        exit 1 ;;
esac

# Collect checkpoint paths for all seeds
CKPT_PATHS=()
for i in $(seq 0 $((NUM_SEEDS - 1))); do
    CKPT=$(ls -td $RESULT_BASE/seed_${i}/*/saved_train_run 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: no checkpoint found for seed $i at $RESULT_BASE/seed_${i}/*/saved_train_run"
        exit 1
    fi
    CKPT_PATHS+=("$CKPT")
    echo "  seed $i: $CKPT"
done

echo "=== XP eval: $TASK (${NUM_SEEDS} seeds) ==="

CUDA_VISIBLE_DEVICES="$GPU" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
LD_LIBRARY_PATH="" \
uv run python -m evaluation.run_xp_seeds \
    --task "$EVAL_TASK_CFG" \
    --checkpoints "${CKPT_PATHS[@]}"
