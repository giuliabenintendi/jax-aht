#!/usr/bin/env bash
# JA_CARD_JSD_COEF beta sweep with CONC/FOLLOW/ALIGN zeroed, so JSD is the
# only JA reward term active. The 5-dim partner-card-attention input is
# still injected (it is hard-wired to JA_CARD_ATTN=True; decoupling it from
# the reward terms would require a code change).
#
# 3 vmapped seeds per beta value. Other settings match
# run_card_game_ja_card_sweep.sh so the curves remain comparable apart from
# the deliberate isolation of the JSD term.
#
# Per-agent obs eval video is logged to wandb automatically because
# JA_CARD_ATTN=true (see marl/eval_logging.py).
#
# Usage: ./run_card_game_ja_card_jsd_high.sh <gpu>

GPU="${1:?Usage: ./run_card_game_ja_card_jsd_high.sh <gpu>}"
TIMESTEPS=5e6
NUM_SEEDS=3
JA_VALUES="0.02 0.05 0.1 0.2 0.5"

COMMON="algorithm.NUM_SEEDS=$NUM_SEEDS algorithm.TRAIN_SEED=42 \
algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
algorithm.COMMUNICATION=true algorithm.JA_BETA_MAX=0.0 \
algorithm.JA_CARD_ATTN=true \
algorithm.JA_CARD_CONC_COEF=0.0 algorithm.JA_CARD_ALIGN_COEF=0.0 algorithm.JA_CARD_FOLLOW_COEF=0.0 \
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
        label=ja_card_jsd_only_\$JA
    echo \"[\$(date +%H:%M)] Finished JA_CARD_JSD_COEF=\$JA\"
done
echo \"[\$(date +%H:%M)] JA card JSD high-beta sweep complete\"
" > logs/card_game_ja_card_jsd_high.log 2>&1 &

echo "Launched JSD-only JA sweep ($JA_VALUES) on GPU $GPU"
echo "  CONC=ALIGN=FOLLOW=0; only JA_CARD_JSD_COEF active"
echo "  NUM_SEEDS=$NUM_SEEDS per cell, $TIMESTEPS env steps each"
echo "PID: $!  |  log: logs/card_game_ja_card_jsd_high.log"
