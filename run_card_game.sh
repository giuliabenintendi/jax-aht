#!/usr/bin/env bash
# Card game experiments. Edit runs below as needed.
# Usage: ./run_card_game.sh <gpu>

GPU="${1:?Usage: ./run_card_game.sh <gpu>}"

mkdir -p logs

nohup bash -c "

echo \"[\$(date +%H:%M)] filter_cards, b=0.1, LR=8e-4, 5M, 5 seeds\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    algorithm.NUM_SEEDS=5 algorithm.TOTAL_TIMESTEPS=5e6 \
    algorithm.FEED_OTHER_ATTN=true algorithm.JA_BETA_MAX=0.1 \
    algorithm.FILTER_ATTN_CARDS=true algorithm.LR=8e-4 \
    label=filter_cards_5M

echo \"[\$(date +%H:%M)] fixed_partner0, b=0.1, LR=8e-4, 5M, 1 seed\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    algorithm.NUM_SEEDS=1 algorithm.TOTAL_TIMESTEPS=5e6 \
    algorithm.FEED_OTHER_ATTN=true algorithm.JA_BETA_MAX=0.1 \
    algorithm.LR=8e-4 \
    task.ENV_KWARGS.fixed_partner_pos=0 \
    label=fixed0_5M

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game.log 2>&1 &

echo "Launched card-game on GPU $GPU (check logs/card_game.log)"
