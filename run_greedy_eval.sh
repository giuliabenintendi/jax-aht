#!/usr/bin/env bash
# Run greedy + stochastic eval on all high-entropy checkpoints,
# uploading results to the existing wandb runs.
#
# Usage:
#   ./run_greedy_eval.sh <gpu>
# Example:
#   ./run_greedy_eval.sh 5

set -e

GPU="${1:?Usage: ./run_greedy_eval.sh <gpu>}"

mkdir -p logs

# checkpoint_path wandb_run_id
RUNS=(
    "results/overcooked-v1/cramped_room/ja_ippo/dual_cr_b025_ent045/2026-03-17_14-30-34/saved_train_run zkk7lyk4"
    "results/overcooked-v1/cramped_room/ja_ippo/dual_cr_b025_jsdgae_ent045/2026-03-17_20-28-47/saved_train_run ze45biub"
    "results/overcooked-v1/coord_ring/ja_ippo/dual_cring_b0001_ent045/2026-03-17_14-33-58/saved_train_run gp5w42fm"
    "results/overcooked-v1/coord_ring/ja_ippo/dual_cring_b0001_jsdgae_ent045/2026-03-17_20-28-47/saved_train_run t0chl2o4"
    "results/overcooked-v1/forced_coord/ja_ippo/dual_fc_b05_ent040/2026-03-17_14-34-39/saved_train_run 8eqsz9te"
    "results/overcooked-v1/forced_coord/ja_ippo/dual_fc_b05_jsdgae_ent040/2026-03-17_20-28-47/saved_train_run 73facnlw"
)

total=${#RUNS[@]}
count=0

echo "=== Greedy eval: $total checkpoints ==="

for entry in "${RUNS[@]}"; do
    CKPT=$(echo "$entry" | awk '{print $1}')
    RUN_ID=$(echo "$entry" | awk '{print $2}')
    count=$((count + 1))
    echo ""
    echo "=== [$count/$total] $CKPT -> $RUN_ID ==="
    ./run_gpu.sh "$GPU" evaluation.eval_greedy \
        --checkpoint "$CKPT" \
        --run-id "$RUN_ID" \
        --num-episodes 256 \
        --output-dir plots/
    echo "Done: $CKPT"
done

echo ""
echo "=== Complete: $count/$total ==="
