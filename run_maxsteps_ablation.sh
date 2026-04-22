#!/usr/bin/env bash
# Deliberation-duration ablation. Runs max_steps={2, 4, 12} sequentially on one GPU.
# TOTAL_TIMESTEPS scales with max_steps to keep ~1M episodes per config.
# Baseline max_steps=8 (5M timesteps = 625K episodes) already covered by a prior run.
#
# Usage: ./run_maxsteps_ablation.sh <gpu>

set -euo pipefail

GPU="${1:?Usage: ./run_maxsteps_ablation.sh <gpu>}"

mkdir -p logs

COMMON="-cn base_config_ja_ippo \
  task=card-game algorithm=ja_ippo/card-game \
  algorithm.COMMUNICATION=true algorithm.JA_BETA_MAX=0.0 \
  algorithm.NUM_SEEDS=7 algorithm.NUM_ENVS=64 algorithm.GAE_LAMBDA=0.95 \
  algorithm.COMM_WARMUP_ENV_STEPS=0 \
  algorithm.LR=7e-4 algorithm.ENT_COEF=0.01 algorithm.ANNEAL_LR=false \
  task.ENV_KWARGS.other_play_position_shuffle=true \
  task.ENV_KWARGS.other_play_recolouring=true \
  task.ENV_KWARGS.match_coef=0.1 \
  task.ENV_KWARGS.stability_coef=0.05 \
  task.ENV_KWARGS.follow_coef=0.5"

nohup bash -c "
echo \"[\$(date +%H:%M)] max_steps=2  (2M timesteps)\"
./run_gpu.sh $GPU marl.run $COMMON \
  algorithm.TOTAL_TIMESTEPS=2e6 \
  task.ENV_KWARGS.max_steps=2 \
  label=maxsteps2_s7

echo \"[\$(date +%H:%M)] max_steps=4  (4M timesteps)\"
./run_gpu.sh $GPU marl.run $COMMON \
  algorithm.TOTAL_TIMESTEPS=4e6 \
  task.ENV_KWARGS.max_steps=4 \
  label=maxsteps4_s7

echo \"[\$(date +%H:%M)] max_steps=12 (12M timesteps)\"
./run_gpu.sh $GPU marl.run $COMMON \
  algorithm.TOTAL_TIMESTEPS=12e6 \
  task.ENV_KWARGS.max_steps=12 \
  label=maxsteps12_s7

echo \"[\$(date +%H:%M)] All three configs finished.\"
" > logs/abl_maxsteps_seq.log 2>&1 &

echo "Launched deliberation-duration ablation on GPU $GPU (sequential: 2 → 4 → 12)."
echo "Log: logs/abl_maxsteps_seq.log"
