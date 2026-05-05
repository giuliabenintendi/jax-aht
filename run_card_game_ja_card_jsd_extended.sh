#!/usr/bin/env bash
# Confirmation run extending ps2w3xd4 (JA-card JSD-only): 12 vmapped seeds,
# 8 M env steps. Same config knobs as ps2w3xd4 — only NUM_SEEDS and
# TOTAL_TIMESTEPS change. The point is to (a) check whether the seed
# stability we saw at n=3 holds at n=12, and (b) confirm the higher ceiling
# vs the comm-warmup baseline lhq6e8wh.
#
# Settings:
#   JA_CARD_ATTN=true, JA_CARD_PARTNER_FEED=false (no input leak)
#   JA_CARD_JSD_COEF=0.1; CONC=ALIGN=FOLLOW=0 (only JSD reward)
#   JA_BETA_MAX=0 (no spatial JSD reward)
#   COMM_WARMUP_ENV_STEPS=0, COMM_REWARD_START_SCALE=0.5 (comm reward live
#     from step 0 at half scale — matches ps2w3xd4)
#
# Note: 12 seeds vmapped × NUM_ENVS=1024 = ~12k parallel envs. Baseline
# lhq6e8wh used the same 12-seed setup and ran fine, so GPU 1 should cope.
# If memory is tight, append `algorithm.NUM_ENVS=512` to the command.
#
# Usage: ./run_card_game_ja_card_jsd_extended.sh <gpu>

GPU="${1:?Usage: ./run_card_game_ja_card_jsd_extended.sh <gpu>}"
NUM_SEEDS=12
TIMESTEPS=8e6

mkdir -p logs

nohup ./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    algorithm.NUM_SEEDS=$NUM_SEEDS algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
    algorithm.COMMUNICATION=true \
    algorithm.JA_BETA_MAX=0.0 \
    algorithm.JA_CARD_ATTN=true algorithm.JA_CARD_PARTNER_FEED=false \
    algorithm.JA_CARD_CONC_COEF=0.0 algorithm.JA_CARD_ALIGN_COEF=0.0 algorithm.JA_CARD_FOLLOW_COEF=0.0 \
    algorithm.JA_CARD_JSD_COEF=0.1 \
    algorithm.GAE_LAMBDA=0.95 algorithm.LR=5e-4 algorithm.ENT_COEF=0.01 algorithm.ANNEAL_LR=false \
    algorithm.COMM_WARMUP_ENV_STEPS=0 algorithm.COMM_REWARD_START_SCALE=0.5 \
    task.ENV_KWARGS.other_play_position_shuffle=true task.ENV_KWARGS.other_play_recolouring=true \
    task.ENV_KWARGS.match_coef=0.1 task.ENV_KWARGS.stability_coef=0.05 task.ENV_KWARGS.follow_coef=0.5 \
    label=ja_card_jsd0.1_12s_8M \
    > logs/card_game_ja_card_jsd_extended.log 2>&1 &

echo "Launched extended JA-card JSD-only run on GPU $GPU"
echo "  NUM_SEEDS=$NUM_SEEDS, $TIMESTEPS env steps, JA_CARD_JSD_COEF=0.1, PARTNER_FEED=false"
echo "PID: $!  |  log: logs/card_game_ja_card_jsd_extended.log"
