#!/usr/bin/env bash
# Relog beta_sweep checkpoints for a layout with clean metrics + videos.
#
# Usage: ./run_relog.sh <gpu_id> <layout> [beta]
#
# beta can be a specific value (e.g. 0.0, 0.1) or "all" (default).
# Matches folder names containing "BETA<value>" (e.g. BETA0.0_seed20374_5M).
#
# Examples:
#   ./run_relog.sh 1 cramped_room           # all betas
#   ./run_relog.sh 1 cramped_room all       # all betas (explicit)
#   ./run_relog.sh 3 forced_coord 0.0       # only BETA=0.0
#   ./run_relog.sh 3 coord_ring 0.5         # only BETA=0.5

set -e

GPU="${1:?Usage: ./run_relog.sh <gpu_id> <layout> [beta]}"
LAYOUT="${2:?Usage: ./run_relog.sh <gpu_id> <layout> [beta]}"
BETA="${3:-all}"

RESULT_DIR="results/overcooked-v1/$LAYOUT/ja_ippo/beta_sweep"

if [ "$BETA" = "all" ]; then
    CKPTS=$(ls -td $RESULT_DIR/*/saved_train_run 2>/dev/null)
else
    CKPTS=$(ls -td $RESULT_DIR/BETA${BETA}*/saved_train_run 2>/dev/null)
fi

if [ -z "$CKPTS" ]; then
    echo "ERROR: no checkpoints found (layout=$LAYOUT, beta=$BETA)"
    echo "  looked in: $RESULT_DIR/"
    exit 1
fi

total=$(echo "$CKPTS" | wc -l | tr -d ' ')
count=0

echo "=== Relog: $LAYOUT, beta=$BETA ($total checkpoints) ==="

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
    nice -n 5 \
    uv run python -m evaluation.relog_stats \
        --checkpoint "$CKPT_PATH"
done

echo ""
echo "=== Relog complete: $count/$total ==="
