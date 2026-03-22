#!/usr/bin/env bash
# Run XP evaluation on all multi-seed checkpoints (overcooked + card-game + lbf).
# Skips checkpoints that already have xp_results/ or have < 2 seeds.
# Usage: ./run_all_xp.sh <gpu>

GPU="${1:?Usage: ./run_all_xp.sh <gpu>}"

mkdir -p logs

CHECKPOINTS=""
for d in results/*/; do
    find "$d" -name "saved_train_run" -type d 2>/dev/null | while read ckpt; do
        run_dir=$(dirname "$ckpt")
        cfg="$run_dir/.hydra/config.yaml"
        [ -f "$cfg" ] || continue
        seeds=$(grep "NUM_SEEDS" "$cfg" 2>/dev/null | awk '{print $2}')
        [ "$seeds" -ge 2 ] 2>/dev/null || continue
        # Skip if XP already done
        [ -d "$run_dir/xp_results" ] && continue
        echo "$ckpt"
    done
done | sort > /tmp/xp_checkpoints.txt

CHECKPOINTS=$(cat /tmp/xp_checkpoints.txt)

if [ -z "$CHECKPOINTS" ]; then
    echo "No new checkpoints to evaluate"
    exit 0
fi

COUNT=$(cat /tmp/xp_checkpoints.txt | wc -l)
echo "Found $COUNT checkpoints for XP eval"
cat /tmp/xp_checkpoints.txt

nohup bash -c "
while read CKPT; do
    echo \"[\$(date +%H:%M)] XP eval: \$CKPT\"
    ./run_gpu.sh $GPU evaluation.run_xp_seeds --checkpoint \$CKPT
done < /tmp/xp_checkpoints.txt
echo \"[\$(date +%H:%M)] All XP evals complete\"
" > logs/all_xp.log 2>&1 &

echo "Launched $COUNT XP evals on GPU $GPU (check logs/all_xp.log)"
