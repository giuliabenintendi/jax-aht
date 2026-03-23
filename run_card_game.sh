#!/usr/bin/env bash
# Card game experiments. Edit runs below as needed.
# Usage: ./run_card_game.sh <gpu>

GPU="${1:?Usage: ./run_card_game.sh <gpu>}"

mkdir -p logs

nohup bash -c "

echo \"[\$(date +%H:%M)] Single critic, beta=1.0, 5M, 5 seeds, no shuffle\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    algorithm.NUM_SEEDS=5 algorithm.TOTAL_TIMESTEPS=5e6 \
    algorithm.FEED_OTHER_ATTN=true algorithm.JA_BETA_MAX=1.0 \
    label=no_shuffle_b1.0_5M

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game.log 2>&1 &

echo "Launched card-game on GPU $GPU (check logs/card_game.log)"
