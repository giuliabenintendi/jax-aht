#!/usr/bin/env bash
# JA + aux-loss + partner-feed sweep at 1/50 of original shaping coefs.
#
# Common setup (all cells):
#   - gaze-only (no comm channel, no comm shaping)
#   - JA_CARD_ATTN=true → partner's canonical card attention as obs input
#   - JA_AUX_PARTNER_ARGMAX_COEF=0.1 → LIAM-style aux loss
#   - OP wrappers on
#   - 6 seeds, 10M steps
#
# Two cells, sequential on GPU 6:
#   1) follow_only:        match=0,     follow=0.01  (safer, no Goodhart risk)
#   2) match_and_follow:   match=0.002, follow=0.01  (small per-step nudge)
#
# Log: sweep_aux.log.

set -u

NUM_SEEDS=6
TOTAL_STEPS=10e6

run_cell() {
  local gpu="$1"
  local match="$2"
  local follow="$3"
  local label="$4"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] starting ${label} on GPU ${gpu}: match=${match} follow=${follow}"
  ./run_gpu.sh "${gpu}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    algorithm.NUM_SEEDS=${NUM_SEEDS} \
    algorithm.TOTAL_TIMESTEPS=${TOTAL_STEPS} \
    algorithm.COMMUNICATION=false \
    task.ENV_KWARGS.match_coef=0.0 \
    task.ENV_KWARGS.stability_coef=0.0 \
    task.ENV_KWARGS.follow_coef=0.0 \
    algorithm.JA_CARD_ATTN=true \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.1 \
    algorithm.JA_ATTN_MATCH_COEF=${match} \
    algorithm.JA_ATTN_FOLLOW_COEF=${follow}
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] finished ${label} (exit ${rc})"
  return ${rc}
}

# Sequential on GPU 6.
(
  run_cell 6 0.0   0.01 "follow_only_1over50"
  run_cell 6 0.002 0.01 "match_and_follow_1over50"
) > sweep_aux.log 2>&1 &
PID=$!

echo "GPU 6 chain PID: ${PID}"
echo "Cells: follow_only_1over50 then match_and_follow_1over50"
echo "Tail log:"
echo "  tail -f sweep_aux.log"
echo "Waiting for the chain to finish..."
wait ${PID}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] All cells finished."
