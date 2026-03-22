#!/usr/bin/env bash
# Run cross-play evaluation on all multi-seed card game checkpoints.
# Usage: ./run_card_game_xp.sh <gpu>

GPU="${1:?Usage: ./run_card_game_xp.sh <gpu>}"

mkdir -p logs

# Find all saved_train_run directories under card-game results
# Skip the 1-seed smoke test run (card-game_ja_ippo_2M_b0.05_feed_attn_21032026)
CHECKPOINTS=$(find results/card-game -name "saved_train_run" -type d 2>/dev/null)

if [ -z "$CHECKPOINTS" ]; then
    echo "No card-game checkpoints found under results/card-game/"
    exit 1
fi

echo "Found checkpoints:"
echo "$CHECKPOINTS"
echo ""

nohup bash -c "
for CKPT in $CHECKPOINTS; do
    echo \"[\$(date +%H:%M)] XP eval: \$CKPT\"
    ./run_gpu.sh $GPU evaluation.run_xp_seeds --checkpoint \$CKPT
done
echo \"[\$(date +%H:%M)] All card-game XP evals complete\"
" > logs/card_game_xp.log 2>&1 &

echo "Launched card-game XP evals on GPU $GPU (check logs/card_game_xp.log)"
