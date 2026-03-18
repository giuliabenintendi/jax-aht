#!/usr/bin/env bash
# Run greedy + stochastic eval on all high-entropy checkpoints.
#
# Usage:
#   ./run_greedy_eval.sh <gpu>
# Example:
#   ./run_greedy_eval.sh 5

set -e

GPU="${1:?Usage: ./run_greedy_eval.sh <gpu>}"

mkdir -p logs

CHECKPOINTS=(
    "results/overcooked-v1/cramped_room/ja_ippo/dual_cr_b025_ent045/2026-03-17_14-30-34/saved_train_run"
    "results/overcooked-v1/cramped_room/ja_ippo/dual_cr_b025_jsdgae_ent045/2026-03-17_20-28-47/saved_train_run"
    "results/overcooked-v1/coord_ring/ja_ippo/dual_cring_b0001_ent045/2026-03-17_14-33-58/saved_train_run"
    "results/overcooked-v1/coord_ring/ja_ippo/dual_cring_b0001_jsdgae_ent045/2026-03-17_20-28-47/saved_train_run"
    "results/overcooked-v1/forced_coord/ja_ippo/dual_fc_b05_ent040/2026-03-17_14-34-39/saved_train_run"
    "results/overcooked-v1/forced_coord/ja_ippo/dual_fc_b05_jsdgae_ent040/2026-03-17_20-28-47/saved_train_run"
)

total=${#CHECKPOINTS[@]}
count=0

echo "=== Greedy eval: $total checkpoints ==="

for CKPT in "${CHECKPOINTS[@]}"; do
    count=$((count + 1))
    echo ""
    echo "=== [$count/$total] $CKPT ==="
    nohup ./run_gpu.sh "$GPU" evaluation.eval_greedy \
        --checkpoint "$CKPT" \
        --num-episodes 256 \
        --output-dir plots/ \
        > "logs/greedy_eval_${count}.log" 2>&1
    echo "Done: $CKPT"
done

echo ""
echo "=== Complete: $count/$total ==="
