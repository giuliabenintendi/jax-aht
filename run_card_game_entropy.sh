#!/usr/bin/env bash
# Card game entropy sweep: dual critic, JSD GAE off, ent=0.02 and 0.04.
# All runs use FEED_OTHER_ATTN=true, 1M timesteps, 5 seeds.
# Usage: ./run_card_game_entropy.sh <gpu>

GPU="${1:?Usage: ./run_card_game_entropy.sh <gpu>}"
SEEDS=5
TIMESTEPS=2e6
COMMON="algorithm.NUM_SEEDS=$SEEDS algorithm.TOTAL_TIMESTEPS=$TIMESTEPS algorithm.FEED_OTHER_ATTN=true"
DUAL="algorithm.USE_DUAL_CRITIC=true algorithm.DUAL_CRITIC_ACTOR_JA=false"

mkdir -p logs

nohup bash -c "

# --- Dual critic, JSD GAE off, entropy=0.02 ---

for BETA in 0.0 0.001 0.01 0.05; do
    echo \"[\$(date +%H:%M)] Dual critic, beta=\$BETA, ent=0.02\"
    ./run_gpu.sh $GPU marl.run \
        -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
        $COMMON $DUAL algorithm.JA_BETA_MAX=\$BETA algorithm.ENT_COEF=0.02 \
        label=shuffled_cards
done

# --- Dual critic, JSD GAE off, entropy=0.04 ---

for BETA in 0.0 0.001 0.01 0.05; do
    echo \"[\$(date +%H:%M)] Dual critic, beta=\$BETA, ent=0.04\"
    ./run_gpu.sh $GPU marl.run \
        -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
        $COMMON $DUAL algorithm.JA_BETA_MAX=\$BETA algorithm.ENT_COEF=0.04 \
        label=shuffled_cards
done

echo \"[\$(date +%H:%M)] All entropy sweep runs complete\"
" > logs/card_game_entropy.log 2>&1 &

echo "Launched 8 entropy sweep experiments on GPU $GPU (check logs/card_game_entropy.log)"
