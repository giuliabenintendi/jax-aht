#!/usr/bin/env bash
# JA-only 3D factorial sweep with all shaping rewards ON.
#
# Shared (all cells):
#   - COMMUNICATION=false, gaze_mode=true, per_head=true
#   - JA_CARD_ATTN=true   -> partner-feed pathway on
#   - 6 seeds, 5M steps   -> ~2h/cell
#
# Grid axes:
#   match  in {0.01, 0.05}        — dense per-step JSD signal during deliberation
#   follow in {0.5, 1.0, 2.0}     — per-agent decision-step pick-matches-attn
#   aux    in {0.1, 0.5}          — LIAM aux loss predicting partner's argmax
# Total: 2 x 3 x 2 = 12 cells. Split across GPU 2 (match=0.01) and GPU 6 (match=0.05).
# Estimated wall time: ~12h overnight.

set -u

NUM_SEEDS=6
TOTAL_STEPS=5e6

run_cell() {
  local gpu="$1"
  local match="$2"
  local follow="$3"
  local aux="$4"
  local label="$5"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} GPU ${gpu} (m=${match} f=${follow} a=${aux})"
  ./run_gpu.sh "${gpu}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    algorithm.NUM_SEEDS=${NUM_SEEDS} \
    algorithm.TOTAL_TIMESTEPS=${TOTAL_STEPS} \
    algorithm.COMMUNICATION=false \
    task.ENV_KWARGS.match_coef=0.0 \
    task.ENV_KWARGS.stability_coef=0.0 \
    task.ENV_KWARGS.follow_coef=0.0 \
    task.ENV_KWARGS.gaze_mode=true \
    algorithm.JA_CARD_ATTN=true \
    algorithm.JA_PARTNER_FEED_PER_HEAD=true \
    algorithm.JA_ATTN_MATCH_COEF=${match} \
    algorithm.JA_ATTN_FOLLOW_COEF=${follow} \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=${aux}
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit ${rc})"
  return ${rc}
}

# GPU 2: match=0.01, sweep follow x aux (6 cells)
(
  run_cell 2 0.01 0.5 0.1  "m001_f05_a01"
  run_cell 2 0.01 0.5 0.5  "m001_f05_a05"
  run_cell 2 0.01 1.0 0.1  "m001_f10_a01"
  run_cell 2 0.01 1.0 0.5  "m001_f10_a05"
  run_cell 2 0.01 2.0 0.1  "m001_f20_a01"
  run_cell 2 0.01 2.0 0.5  "m001_f20_a05"
) > grid_rich_gpu2.log 2>&1 &
PID_2=$!

# GPU 6: match=0.05, sweep follow x aux (6 cells)
(
  run_cell 6 0.05 0.5 0.1  "m005_f05_a01"
  run_cell 6 0.05 0.5 0.5  "m005_f05_a05"
  run_cell 6 0.05 1.0 0.1  "m005_f10_a01"
  run_cell 6 0.05 1.0 0.5  "m005_f10_a05"
  run_cell 6 0.05 2.0 0.1  "m005_f20_a01"
  run_cell 6 0.05 2.0 0.5  "m005_f20_a05"
) > grid_rich_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 2 chain PID: ${PID_2}  (match=0.01, follow x aux)"
echo "GPU 6 chain PID: ${PID_6}  (match=0.05, follow x aux)"
echo "Logs:"
echo "  tail -f grid_rich_gpu2.log"
echo "  tail -f grid_rich_gpu6.log"
echo "Waiting for both chains to finish..."
wait ${PID_2} ${PID_6}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Grid finished."
