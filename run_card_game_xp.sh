#!/usr/bin/env bash
# Run cross-play evaluation on 2M, 5-seed card game checkpoints.
# Usage: ./run_card_game_xp.sh <gpu>

GPU="${1:?Usage: ./run_card_game_xp.sh <gpu>}"

mkdir -p logs

# Only select checkpoints with NUM_SEEDS=5 and TOTAL_TIMESTEPS=2000000
CHECKPOINTS=""
for d in results/card-game/ja_ippo/shuffled_cards/*/; do
    cfg="$d/.hydra/config.yaml"
    [ -f "$cfg" ] || continue
    seeds=$(grep "NUM_SEEDS" "$cfg" 2>/dev/null | awk '{print $2}')
    ts=$(grep "TOTAL_TIMESTEPS" "$cfg" 2>/dev/null | awk '{print $2}')
    if [ "$seeds" = "5" ] && [ "$ts" = "2000000.0" ]; then
        CHECKPOINTS="$CHECKPOINTS ${d}saved_train_run"
    fi
done

if [ -z "$CHECKPOINTS" ]; then
    echo "No 2M 5-seed card-game checkpoints found"
    exit 1
fi

COUNT=$(echo $CHECKPOINTS | wc -w)
echo "Found $COUNT checkpoints for XP eval"

nohup bash -c "
for CKPT in $CHECKPOINTS; do
    echo \"[\$(date +%H:%M)] XP eval: \$CKPT\"
    ./run_gpu.sh $GPU evaluation.run_xp_seeds --checkpoint \$CKPT
done
echo \"[\$(date +%H:%M)] All card-game XP evals complete\"
" > logs/card_game_xp.log 2>&1 &

echo "Launched $COUNT card-game XP evals on GPU $GPU (check logs/card_game_xp.log)"
