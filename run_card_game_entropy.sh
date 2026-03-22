#!/usr/bin/env bash
# Card game high entropy sweep.
# Usage: ./run_card_game_entropy.sh <gpu>

GPU="${1:?Usage: ./run_card_game_entropy.sh <gpu>}"
SEEDS=5
TIMESTEPS=2e6
COMMON="algorithm.NUM_SEEDS=$SEEDS algorithm.TOTAL_TIMESTEPS=$TIMESTEPS algorithm.FEED_OTHER_ATTN=true"
DUAL="algorithm.USE_DUAL_CRITIC=true algorithm.DUAL_CRITIC_ACTOR_JA=false"

mkdir -p logs

nohup bash -c "

for ENT in 0.1 0.2; do
    for BETA in 0.0 0.001 0.01 0.05 0.1; do
        echo \"[\$(date +%H:%M)] Dual critic, beta=\$BETA, ent=\$ENT\"
        ./run_gpu.sh $GPU marl.run \
            -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
            $COMMON $DUAL algorithm.JA_BETA_MAX=\$BETA algorithm.ENT_COEF=\$ENT \
            label=shuffled_cards
    done
done

echo \"[\$(date +%H:%M)] All high entropy runs complete\"
" > logs/card_game_entropy.log 2>&1 &

echo "Launched 10 card-game high entropy runs on GPU $GPU (check logs/card_game_entropy.log)"
