#!/usr/bin/env bash
# Card game experiments. Edit runs below as needed.
# Usage: ./run_card_game.sh <gpu>

GPU="${1:?Usage: ./run_card_game.sh <gpu>}"

mkdir -p logs

nohup bash -c "

echo \"[\$(date +%H:%M)] comm25 b0.1, LR=8e-4, 2M, 5 seeds\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    algorithm.NUM_SEEDS=5 algorithm.TOTAL_TIMESTEPS=2e6 \
    algorithm.COMMUNICATION=true \
    algorithm.FEED_OTHER_ATTN=true algorithm.JA_BETA_MAX=0.1 \
    algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=1400000 \
    label=comm25_b0.1_2M_s5

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game.log 2>&1 &

echo "Launched comm25 card-game run on GPU $GPU (check logs/card_game.log)"
