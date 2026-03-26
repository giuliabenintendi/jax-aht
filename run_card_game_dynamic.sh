#!/usr/bin/env bash
# Dynamic card game experiments.
# Usage: ./run_card_game_dynamic.sh <gpu>

GPU="${1:?Usage: ./run_card_game_dynamic.sh <gpu>}"

mkdir -p logs

nohup bash -c "

echo \"[\$(date +%H:%M)] dyn10 6x10 ent=0.2 beta=0.5, 250k, 6 seeds\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game-dynamic algorithm=ja_ippo/card-game \
    algorithm.NUM_SEEDS=6 algorithm.TOTAL_TIMESTEPS=2.5e5 \
    algorithm.FEED_OTHER_ATTN=true algorithm.JA_BETA_MAX=0.5 \
    algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=175000 \
    algorithm.ENT_COEF=0.2 \
    label=dyn10_6x10_ent0.2_b0.5_s6_250k

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_dynamic.log 2>&1 &

echo "Launched dynamic 10c 6x10 run on GPU $GPU (check logs/card_game_dynamic.log)"
