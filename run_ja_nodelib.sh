#!/usr/bin/env bash
# No-deliberation-action launcher (task=card-game-op-nodelib, gaze_mode=true).
# Deliberation steps emit no action (forced noop slot); the decision-step pick
# is the only real action. self/gaze_pick fire only at the decision step;
# match fires per deliberation step. Coefficients via env vars (see
# grid_ja_nodelib.sh).
set -u

GPU="${1:-0}"
MATCH="${MATCH:-0.0}"
SELF="${SELF:-0.05}"
GAZE_PICK="${GAZE_PICK:-0.30}"
AUX="${AUX:-0.25}"
SEEDS="${SEEDS:-1}"
STEPS="${STEPS:-15e6}"
PER_HEAD="${PER_HEAD:-false}"

LABEL="${LABEL:-nodelib_match${MATCH}_self${SELF}_gaze${GAZE_PICK}_aux${AUX}}"

echo "[$(date +%Y-%m-%d_%H:%M:%S)] starting ${LABEL} on GPU ${GPU}"

./run_gpu.sh "${GPU}" marl.run \
  task=card-game-op-nodelib \
  algorithm=ja_ippo/card-game-op-nodelib \
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
