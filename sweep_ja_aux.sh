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
# Two cells, in parallel:
#   GPU 5: follow-only at 1/50         (match=0,     follow=0.01)
#   GPU 6: match+follow at 1/50        (match=0.002, follow=0.01)
#
# The follow-only cell tests whether the safer task-grounded shaping is
# enough alongside the aux loss. The match+follow cell tests whether the
# extra per-step match nudge helps without re-triggering Goodhart.
#
# Logs: sweep_aux_gpu5.log, sweep_aux_gpu6.log.

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

# GPU 5: follow-only (no match shaping). Safest, no Goodhart risk.
(
  run_cell 5 0.0 0.01 "follow_only_1over50"
) > sweep_aux_gpu5.log 2>&1 &
PID_5=$!

# GPU 6: match + follow at 1/50 original.
(
  run_cell 6 0.002 0.01 "match_and_follow_1over50"
) > sweep_aux_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 5 chain PID: ${PID_5} (follow_only_1over50)"
echo "GPU 6 chain PID: ${PID_6} (match_and_follow_1over50)"
echo "Tail logs:"
echo "  tail -f sweep_aux_gpu5.log"
echo "  tail -f sweep_aux_gpu6.log"
echo "Waiting for both chains to finish..."
wait ${PID_5} ${PID_6}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] All cells finished."
