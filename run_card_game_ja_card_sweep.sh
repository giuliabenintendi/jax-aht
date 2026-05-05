#!/usr/bin/env bash
# Card-level joint attention beta search on a single seed.
# Card-level JA computes JSD over per-card attention logits in ground-truth color
# space, which is the correct JA variant under OP recolouring (spatial JSD is
# meaningless because agents see different recoloured images).
#
# Sweep values are fractions of match_coef=0.1: {1/500, 1/100, 1/50, 1/10}
#   = {0.0002, 0.001, 0.002, 0.01}
# Single seed (TRAIN_SEED=42) per cell so we can identify a promising beta cheaply
# (~30min per cell, 4 cells -> ~2h total). Promote the winner to multi-seed afterwards.
# Usage: ./run_card_game_ja_card_sweep.sh <gpu>

GPU="${1:?Usage: ./run_card_game_ja_card_sweep.sh <gpu>}"
TIMESTEPS=5e6
JA_VALUES="0.0002 0.001 0.002 0.01"

COMMON="algorithm.NUM_SEEDS=1 algorithm.TRAIN_SEED=42 algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
algorithm.COMMUNICATION=true algorithm.JA_BETA_MAX=0.0 \
algorithm.JA_CARD_ATTN=true \
algorithm.GAE_LAMBDA=0.95 algorithm.LR=5e-4 algorithm.ENT_COEF=0.01 algorithm.ANNEAL_LR=false \
algorithm.COMM_WARMUP_ENV_STEPS=0 \
task.ENV_KWARGS.other_play_position_shuffle=true task.ENV_KWARGS.other_play_recolouring=true \
task.ENV_KWARGS.match_coef=0.1 task.ENV_KWARGS.stability_coef=0.05 task.ENV_KWARGS.follow_coef=0.5"

mkdir -p logs

nohup bash -c "
for JA in $JA_VALUES; do
    echo \"[\$(date +%H:%M)] Starting JA_CARD_JSD_COEF=\$JA\"
    ./run_gpu.sh $GPU marl.run \
        -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
        $COMMON algorithm.JA_CARD_JSD_COEF=\$JA \
        label=ja_card_sweep_jsd\$JA
    echo \"[\$(date +%H:%M)] Finished JA_CARD_JSD_COEF=\$JA\"
done
echo \"[\$(date +%H:%M)] JA card-level sweep complete\"
" > logs/card_game_ja_card_sweep.log 2>&1 &

echo "Launched JA_CARD_JSD_COEF sweep ($JA_VALUES) on GPU $GPU"
echo "PID: $!  |  log: logs/card_game_ja_card_sweep.log"
