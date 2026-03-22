#!/usr/bin/env bash
# Card game single critic high beta sweep.
# Usage: ./run_card_game.sh <gpu>

GPU="${1:?Usage: ./run_card_game.sh <gpu>}"
SEEDS=5
TIMESTEPS=2e6
COMMON="algorithm.NUM_SEEDS=$SEEDS algorithm.TOTAL_TIMESTEPS=$TIMESTEPS algorithm.FEED_OTHER_ATTN=true"

mkdir -p logs

nohup bash -c "

for BETA in 0.25 0.5; do
    echo \"[\$(date +%H:%M)] Single critic, beta=\$BETA\"
    ./run_gpu.sh $GPU marl.run \
        -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
        $COMMON algorithm.JA_BETA_MAX=\$BETA \
        label=shuffled_cards
done

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game.log 2>&1 &

echo "Launched card-game high beta runs on GPU $GPU (check logs/card_game.log)"
