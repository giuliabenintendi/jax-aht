#!/usr/bin/env bash
# Quick Other-Play sanity run for the static card game communication setup.
# Usage: ./run_card_game_op_smoke.sh <gpu> [match_coef] [timesteps]

set -euo pipefail

GPU="${1:?Usage: ./run_card_game_op_smoke.sh <gpu> [match_coef] [timesteps]}"
MATCH_COEF="${2:-0.05}"
TOTAL_TIMESTEPS="${3:-100000}"
WARMUP_STEPS=$((TOTAL_TIMESTEPS / 2))

LABEL="op_comm_smoke_m${MATCH_COEF}_t${TOTAL_TIMESTEPS}"

echo "Running static card-game OP smoke test"
echo "  gpu: ${GPU}"
echo "  match_coef: ${MATCH_COEF}"
echo "  total_timesteps: ${TOTAL_TIMESTEPS}"
echo "  label: ${LABEL}"

./run_gpu.sh "${GPU}" marl.run -cn base_config_ja_ippo \
  task=card-game \
  algorithm=ja_ippo_no_share/card-game \
  algorithm.NUM_SEEDS=1 \
  algorithm.NUM_ENVS=32 \
  algorithm.TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
  algorithm.LR=8e-4 \
  algorithm.JA_WARMUP_ENV_STEPS="${WARMUP_STEPS}" \
  algorithm.COMMUNICATION=true \
  algorithm.FEED_OTHER_ATTN=true \
  algorithm.JA_BETA_MAX=0.0 \
  algorithm.ENT_COEF=0.1 \
  task.ENV_KWARGS.other_play_position_shuffle=true \
  task.ENV_KWARGS.other_play_recolouring=true \
  task.ENV_KWARGS.match_coef="${MATCH_COEF}" \
  label="${LABEL}"
