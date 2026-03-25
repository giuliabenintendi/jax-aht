#!/usr/bin/env bash
# Card game experiments. Edit runs below as needed.
# Usage: ./run_card_game.sh <gpu>

GPU="${1:?Usage: ./run_card_game.sh <gpu>}"

mkdir -p logs

nohup bash -c "

echo \"[\$(date +%H:%M)] comm30 b0.5, ent0.08, LR=8e-4, 1M, 6 seeds\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    algorithm.NUM_SEEDS=6 algorithm.TOTAL_TIMESTEPS=1e6 \
    algorithm.COMMUNICATION=true \
    algorithm.FEED_OTHER_ATTN=true algorithm.JA_BETA_MAX=0.25 \
    algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=700000 \
    algorithm.ENT_COEF=0.08 \
    label=comm30_b0.25_ent0.08_s6_1M

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game.log 2>&1 &

echo "Launched comm30 ent0.08 card-game run on GPU $GPU (check logs/card_game.log)"
