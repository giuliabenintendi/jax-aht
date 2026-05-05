#!/usr/bin/env bash
# Card-level joint attention sweep, layered on top of the working comm+OP setup
# that produced legendary-midichlorian-1245 (SP=0.84 / XP=0.79 over 12 seeds).
#
# Card-level JA computes JSD over per-card attention logits in ground-truth color
# space, which is the correct JA variant under OP recolouring (spatial JSD is
# meaningless because agents see different recoloured images).
#
# Sweeps JA_CARD_JSD_COEF over a small grid; everything else matches the baseline.
# 12 seeds per cell, 5M timesteps each, sequential.
# Usage: ./run_card_game_ja_card_sweep.sh <gpu>

GPU="${1:?Usage: ./run_card_game_ja_card_sweep.sh <gpu>}"
TIMESTEPS=5e6
JA_VALUES="0.01 0.05 0.1 0.2"

COMMON="algorithm.NUM_SEEDS=12 algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
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
