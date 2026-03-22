#!/usr/bin/env bash
# Run cross-play evaluation on multi-seed card game checkpoints.
# Skips checkpoints that already have xp_results/.
# Usage: ./run_card_game_xp.sh <gpu>

GPU="${1:?Usage: ./run_card_game_xp.sh <gpu>}"

mkdir -p logs

CHECKPOINTS=""
for d in results/card-game/ja_ippo/*/; do
    for run_dir in "$d"*/; do
        cfg="$run_dir/.hydra/config.yaml"
        ckpt="$run_dir/saved_train_run"
        [ -f "$cfg" ] || continue
        [ -d "$ckpt" ] || continue
        seeds=$(grep "NUM_SEEDS" "$cfg" 2>/dev/null | awk '{print $2}')
        [ "$seeds" = "5" ] || continue
        # Skip if XP already done
        [ -d "$run_dir/xp_results" ] && continue
        CHECKPOINTS="$CHECKPOINTS $ckpt"
    done
done

if [ -z "$CHECKPOINTS" ]; then
    echo "No new checkpoints to evaluate"
    exit 0
fi

COUNT=$(echo $CHECKPOINTS | wc -w)
echo "Found $COUNT checkpoints for XP eval (skipped already-evaluated ones)"

nohup bash -c "
for CKPT in $CHECKPOINTS; do
    echo \"[\$(date +%H:%M)] XP eval: \$CKPT\"
    ./run_gpu.sh $GPU evaluation.run_xp_seeds --checkpoint \$CKPT
done
echo \"[\$(date +%H:%M)] All card-game XP evals complete\"
" > logs/card_game_xp.log 2>&1 &

echo "Launched $COUNT card-game XP evals on GPU $GPU (check logs/card_game_xp.log)"
