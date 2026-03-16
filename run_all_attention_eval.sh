#!/usr/bin/env bash
# Run multi-episode attention eval on all multi-seed checkpoints.
# Skips checkpoints with fewer than 2 seeds (same filter as run_all_xp.sh).
#
# Usage: ./run_all_attention_eval.sh <gpu_id> [layout] [num_episodes]
#
# Examples:
#   ./run_all_attention_eval.sh 5                    # all layouts, 64 episodes
#   ./run_all_attention_eval.sh 5 cramped_room       # one layout only
#   ./run_all_attention_eval.sh 5 all 32             # all layouts, 32 episodes

set -e

GPU="${1:?Usage: ./run_all_attention_eval.sh <gpu_id> [layout] [num_episodes]}"
LAYOUT="${2:-all}"
NUM_EPS="${3:-64}"

BASE_DIR="results/overcooked-v1"

if [ "$LAYOUT" = "all" ]; then
    SEARCH_DIR="$BASE_DIR"
else
    SEARCH_DIR="$BASE_DIR/$LAYOUT"
fi

CKPTS=$(find "$SEARCH_DIR" -path "*/saved_train_run" -type d 2>/dev/null | sort)
if [ -z "$CKPTS" ]; then
    echo "ERROR: no checkpoints found under $SEARCH_DIR"
    exit 1
fi

total=$(echo "$CKPTS" | wc -l | tr -d ' ')
count=0
skipped=0

echo "=== Attention eval: layout=$LAYOUT, episodes=$NUM_EPS ($total checkpoints) ==="

for CKPT_PATH in $CKPTS; do
    RUN_DIR=$(dirname "$CKPT_PATH")

    # Skip single-seed checkpoints (check num_seeds from saved params)
    NUM_SEEDS=$(CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python -c "
from common.save_load_utils import load_train_run
import jax
out = load_train_run('$CKPT_PATH')
n = jax.tree.leaves(out['final_params'])[0].shape[0]
print(n)
" 2>/dev/null)

    if [ -z "$NUM_SEEDS" ] || [ "$NUM_SEEDS" -lt 2 ]; then
        skipped=$((skipped + 1))
        continue
    fi

    count=$((count + 1))
    RUN_NAME=$(basename "$RUN_DIR")
    LAYOUT_NAME=$(echo "$CKPT_PATH" | sed "s|$BASE_DIR/||" | cut -d/ -f1)

    echo ""
    echo "=== [$count] $LAYOUT_NAME / $RUN_NAME (seeds=$NUM_SEEDS) ==="

    CUDA_VISIBLE_DEVICES="$GPU" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    LD_LIBRARY_PATH="" \
    nice -n 5 \
    uv run python -m evaluation.eval_attention_multi \
        "$RUN_DIR" \
        --num-episodes "$NUM_EPS"
done

echo ""
echo "=== Attention eval complete: $count processed, $skipped skipped (single-seed) ==="
