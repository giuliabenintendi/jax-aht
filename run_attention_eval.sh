#!/usr/bin/env bash
# Compute attention stasis + object coverage for all Overcooked checkpoints.
#
# Usage:
#   ./run_attention_eval.sh <gpu>
# Example:
#   ./run_attention_eval.sh 6

set -e

GPU="${1:?Usage: ./run_attention_eval.sh <gpu>}"

ALL_CKPTS=$(find results/overcooked-v1 -path "*/ja_ippo/beta_sweep/*/saved_train_run" -type d 2>/dev/null)

if [ -z "$ALL_CKPTS" ]; then
    echo "ERROR: no checkpoints found"
    exit 1
fi

total=$(echo "$ALL_CKPTS" | wc -l)
count=0

echo "=== Attention eval: $total checkpoints ==="

for CKPT in $ALL_CKPTS; do
    count=$((count + 1))
    echo ""
    echo "=== [$count/$total] $CKPT ==="
    ./run_gpu.sh "$GPU" evaluation.compute_attention_metrics \
        --checkpoint "$CKPT"
done

echo ""
echo "=== Complete: $count/$total ==="
