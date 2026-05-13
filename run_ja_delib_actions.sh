#!/usr/bin/env bash
set -u

GPU="${1:-0}"
MATCH="${MATCH:-0.05}"
SELF="${SELF:-0.10}"
GAZE_PICK="${GAZE_PICK:-0.0}"
AUX="${AUX:-0.0}"
SEEDS="${SEEDS:-5}"
STEPS="${STEPS:-5e6}"
PER_HEAD="${PER_HEAD:-false}"

LABEL="${LABEL:-match${MATCH}_self${SELF}_gaze${GAZE_PICK}_aux${AUX}}"

echo "[$(date +%Y-%m-%d_%H:%M:%S)] starting ${LABEL} on GPU ${GPU}"

./run_gpu.sh "${GPU}" marl.run \
  task=card-game-op-delib-actions \
  algorithm=ja_ippo/card-game-op-delib-actions \
  label="${LABEL}" \
  algorithm.NUM_SEEDS="${SEEDS}" \
  algorithm.TOTAL_TIMESTEPS="${STEPS}" \
  algorithm.JA_ATTN_MATCH_COEF="${MATCH}" \
  algorithm.JA_ATTN_SELF_COEF="${SELF}" \
  algorithm.JA_GAZE_PICK_COEF="${GAZE_PICK}" \
  algorithm.JA_AUX_PARTNER_ARGMAX_COEF="${AUX}" \
  algorithm.JA_PARTNER_FEED_PER_HEAD="${PER_HEAD}"

rc=$?
echo "[$(date +%Y-%m-%d_%H:%M:%S)] finished ${LABEL} (exit ${rc})"
exit ${rc}
