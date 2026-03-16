#!/usr/bin/env bash
# Run multi-episode attention eval on all multi-seed beta_sweep checkpoints.
#
# Usage: ./run_all_attention_eval.sh <gpu_id> [num_episodes]
#
# Examples:
#   ./run_all_attention_eval.sh 5       # 64 episodes per seed
#   ./run_all_attention_eval.sh 5 16    # 16 episodes (faster)

set -e

GPU="${1:?Usage: ./run_all_attention_eval.sh <gpu_id> [num_episodes]}"
NUM_EPS="${2:-64}"

BASE_DIR="results/overcooked-v1"

CKPTS=$(find "$BASE_DIR" -path "*/beta_sweep/*/saved_train_run" -type d 2>/dev/null | sort)
if [ -z "$CKPTS" ]; then
    echo "ERROR: no beta_sweep checkpoints found under $BASE_DIR"
    exit 1
fi

total=$(echo "$CKPTS" | wc -l | tr -d ' ')
count=0

echo "=== Attention eval: beta_sweep, episodes=$NUM_EPS ($total checkpoints) ==="

for CKPT_PATH in $CKPTS; do
    RUN_DIR=$(dirname "$CKPT_PATH")
    count=$((count + 1))
    RUN_NAME=$(basename "$RUN_DIR")
    LAYOUT=$(echo "$CKPT_PATH" | sed "s|$BASE_DIR/||" | cut -d/ -f1)

    # Read beta from Hydra config
    CFG="$RUN_DIR/.hydra/config.yaml"
    BETA=$(grep "JA_BETA_MAX:" "$CFG" 2>/dev/null | awk '{print $2}' || echo "?")
    SEEDS=$(grep "NUM_SEEDS:" "$CFG" 2>/dev/null | head -1 | awk '{print $2}' || echo "?")
    NORM=$(grep "NORMALIZE_REWARDS:" "$CFG" 2>/dev/null | awk '{print $2}' || echo "?")

    echo ""
    echo "=== [$count/$total] $LAYOUT | beta=$BETA | seeds=$SEEDS | norm=$NORM | $RUN_NAME ==="

    ./run_gpu.sh "$GPU" evaluation.eval_attention_multi \
        "$RUN_DIR" \
        --num-episodes "$NUM_EPS"
done

echo ""
echo "=== Attention eval complete: $count/$total ==="
