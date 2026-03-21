#!/usr/bin/env bash
# Card game sweep GPU A: single critic + dual critic ent=0.01.
# Usage: ./run_card_game.sh <gpu>

GPU="${1:?Usage: ./run_card_game.sh <gpu>}"
SEEDS=5
TIMESTEPS=2e6
COMMON="algorithm.NUM_SEEDS=$SEEDS algorithm.TOTAL_TIMESTEPS=$TIMESTEPS algorithm.FEED_OTHER_ATTN=true"
DUAL="algorithm.USE_DUAL_CRITIC=true algorithm.DUAL_CRITIC_ACTOR_JA=false"

mkdir -p logs

nohup bash -c "

# --- Single critic, ent=0.01 ---

for BETA in 0.0 0.001 0.01 0.05 0.1; do
    echo \"[\$(date +%H:%M)] Single critic, beta=\$BETA\"
    ./run_gpu.sh $GPU marl.run \
        -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
        $COMMON algorithm.JA_BETA_MAX=\$BETA \
        label=shuffled_cards
done

# --- Dual critic, JSD GAE off, ent=0.01 ---

for BETA in 0.0 0.001 0.01 0.05 0.1; do
    echo \"[\$(date +%H:%M)] Dual critic, beta=\$BETA, ent=0.01\"
    ./run_gpu.sh $GPU marl.run \
        -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
        $COMMON $DUAL algorithm.JA_BETA_MAX=\$BETA \
        label=shuffled_cards
done

echo \"[\$(date +%H:%M)] GPU A runs complete\"
" > logs/card_game.log 2>&1 &

echo "Launched 8 runs on GPU $GPU (check logs/card_game.log)"
