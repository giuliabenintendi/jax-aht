#!/usr/bin/env bash
# Card game experiments. Edit runs below as needed.
# Usage: ./run_card_game.sh <gpu>

GPU="${1:?Usage: ./run_card_game.sh <gpu>}"

mkdir -p logs

nohup bash -c "

echo \"[\$(date +%H:%M)] comm30 b0.5, LR=8e-4, 500k, 4 seeds\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    algorithm.NUM_SEEDS=4 algorithm.TOTAL_TIMESTEPS=5e5 \
    algorithm.COMMUNICATION=true \
    algorithm.FEED_OTHER_ATTN=true algorithm.JA_BETA_MAX=0.5 \
    algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=250000 \
    label=comm30_b0.5_s4_500k

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game.log 2>&1 &

echo "Launched comm30 card-game run on GPU $GPU (check logs/card_game.log)"
