#!/usr/bin/env bash
# Card game experiments: single vs dual critic, beta sweep.
# All runs use FEED_OTHER_ATTN=true, 1M timesteps, 3 seeds.
# Usage: ./run_card_game.sh <gpu>
# Example: ./run_card_game.sh 0

GPU="${1:?Usage: ./run_card_game.sh <gpu>}"
SEEDS=5
TIMESTEPS=1e6
COMMON="algorithm.NUM_SEEDS=$SEEDS algorithm.TOTAL_TIMESTEPS=$TIMESTEPS algorithm.FEED_OTHER_ATTN=true"

mkdir -p logs

nohup bash -c "

# --- Single critic ---

echo \"[\$(date +%H:%M)] Single critic, beta=0\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    $COMMON algorithm.JA_BETA_MAX=0.0 \
    label=shuffled_cards

echo \"[\$(date +%H:%M)] Single critic, beta=0.001\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    $COMMON algorithm.JA_BETA_MAX=0.001 \
    label=shuffled_cards

echo \"[\$(date +%H:%M)] Single critic, beta=0.01\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    $COMMON algorithm.JA_BETA_MAX=0.01 \
    label=shuffled_cards

echo \"[\$(date +%H:%M)] Single critic, beta=0.05\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    $COMMON algorithm.JA_BETA_MAX=0.05 \
    label=shuffled_cards

# --- Dual critic, JSD GAE off ---

echo \"[\$(date +%H:%M)] Dual critic (no JSD GAE), beta=0\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    $COMMON algorithm.JA_BETA_MAX=0.0 \
    algorithm.USE_DUAL_CRITIC=true algorithm.DUAL_CRITIC_ACTOR_JA=false \
    label=shuffled_cards

echo \"[\$(date +%H:%M)] Dual critic (no JSD GAE), beta=0.001\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    $COMMON algorithm.JA_BETA_MAX=0.001 \
    algorithm.USE_DUAL_CRITIC=true algorithm.DUAL_CRITIC_ACTOR_JA=false \
    label=shuffled_cards

echo \"[\$(date +%H:%M)] Dual critic (no JSD GAE), beta=0.01\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    $COMMON algorithm.JA_BETA_MAX=0.01 \
    algorithm.USE_DUAL_CRITIC=true algorithm.DUAL_CRITIC_ACTOR_JA=false \
    label=shuffled_cards

echo \"[\$(date +%H:%M)] Dual critic (no JSD GAE), beta=0.05\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    $COMMON algorithm.JA_BETA_MAX=0.05 \
    algorithm.USE_DUAL_CRITIC=true algorithm.DUAL_CRITIC_ACTOR_JA=false \
    label=shuffled_cards

echo \"[\$(date +%H:%M)] All card-game runs complete\"
" > logs/card_game.log 2>&1 &

echo "Launched 8 card-game experiments on GPU $GPU (check logs/card_game.log)"
