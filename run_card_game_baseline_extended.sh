#!/usr/bin/env bash
# Extended baseline (no JA card attention) at 8 seeds and 8 M env steps with
# the matched 50% comm-reward warmup (4 M). Intended as a fair comparison
# point for the 12-seed JA card JSD-only extended run (ja_card_jsd0.1_12s_8M)
# and for the 8-seed max_steps cells in the episode-length ablation.
#
# Settings (baseline, no JA_CARD_ATTN):
#   JA_BETA_MAX=0, JA_CARD_ATTN=false (default)
#   COMM_WARMUP_ENV_STEPS=4M, COMM_REWARD_START_SCALE=0 (linear ramp 0→1
#     over the first half of training)
#   NUM_SEEDS=8, TOTAL_TIMESTEPS=8M
#
# Usage: ./run_card_game_baseline_extended.sh <gpu>

GPU="${1:?Usage: ./run_card_game_baseline_extended.sh <gpu>}"
NUM_SEEDS=8
TIMESTEPS=8e6
WARMUP=4000000

mkdir -p logs

nohup ./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    algorithm.NUM_SEEDS=$NUM_SEEDS algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
    algorithm.COMMUNICATION=true \
    algorithm.JA_BETA_MAX=0.0 \
    algorithm.GAE_LAMBDA=0.95 algorithm.LR=5e-4 algorithm.ENT_COEF=0.01 algorithm.ANNEAL_LR=false \
    algorithm.COMM_WARMUP_ENV_STEPS=$WARMUP algorithm.COMM_REWARD_START_SCALE=0.0 \
    task.ENV_KWARGS.other_play_position_shuffle=true task.ENV_KWARGS.other_play_recolouring=true \
    task.ENV_KWARGS.match_coef=0.1 task.ENV_KWARGS.stability_coef=0.05 task.ENV_KWARGS.follow_coef=0.5 \
    label=baseline_8s_8M_warmup4M \
    > logs/card_game_baseline_extended.log 2>&1 &

echo "Launched extended baseline on GPU $GPU"
echo "  NUM_SEEDS=$NUM_SEEDS, $TIMESTEPS env steps, COMM_WARMUP=$WARMUP, JA_CARD_ATTN=false"
echo "PID: $!  |  log: logs/card_game_baseline_extended.log"
