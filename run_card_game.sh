#!/usr/bin/env bash
# Card game experiments. Edit runs below as needed.
# Usage: ./run_card_game.sh <gpu>

GPU="${1:?Usage: ./run_card_game.sh <gpu>}"
SEEDS=5
TIMESTEPS=2e6
COMMON="algorithm.NUM_SEEDS=$SEEDS algorithm.TOTAL_TIMESTEPS=$TIMESTEPS algorithm.FEED_OTHER_ATTN=true"

mkdir -p logs

nohup bash -c "

echo \"[\$(date +%H:%M)] Single critic, beta=0.01, rollout=128\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    $COMMON algorithm.JA_BETA_MAX=0.01 \
    label=rollout128

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game.log 2>&1 &

echo "Launched card-game test on GPU $GPU (check logs/card_game.log)"
