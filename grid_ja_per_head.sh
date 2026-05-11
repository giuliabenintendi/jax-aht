#!/usr/bin/env bash
# Grid sweep with per-head partner-feed (20-dim) + dense match signal.
#
# Match is the DENSE per-step signal we need. With per-head feed and aux
# loss now in place, we test whether match magnitudes that previously
# Goodhart-farmed (0.1) can be useful again.
#
# Shared config (all cells):
#   - gaze-only (no comm, no comm shaping)
#   - JA_CARD_ATTN=true → partner-feed input pathway on
#   - JA_PARTNER_FEED_PER_HEAD=true → 20-dim per-head feed
#   - JA_AUX_PARTNER_ARGMAX_COEF=0.1 (aux on, fixed)
#   - OP wrappers on
#   - 6 seeds, 10M steps per cell  (~1 hour/cell)
#
# Grid (8 cells, 2 axes):
#   match  ∈ {0.005, 0.01, 0.05, 0.1}
#   follow ∈ {0.5, 1.0}
#
# Parallel across GPU 2 and GPU 6:
#   GPU 2: follow=0.5, all 4 match values
#   GPU 6: follow=1.0, all 4 match values
# Total wall time: ~4 hours per GPU.

set -u

NUM_SEEDS=6
TOTAL_STEPS=10e6

run_cell() {
  local gpu="$1"
  local match="$2"
  local follow="$3"
  local label="$4"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} on GPU ${gpu} (match=${match}, follow=${follow})"
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
    algorithm.JA_PARTNER_FEED_PER_HEAD=true \
    algorithm.JA_ATTN_MATCH_COEF=${match} \
    algorithm.JA_ATTN_FOLLOW_COEF=${follow} \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.1
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit ${rc})"
  return ${rc}
}

# GPU 2: follow=0.5, sweep match
(
  run_cell 2 0.005 0.5 "m0005_f05"
  run_cell 2 0.01  0.5 "m001_f05"
  run_cell 2 0.05  0.5 "m005_f05"
  run_cell 2 0.1   0.5 "m01_f05"
) > grid_per_head_gpu2.log 2>&1 &
PID_2=$!

# GPU 6: follow=1.0, sweep match
(
  run_cell 6 0.005 1.0 "m0005_f10"
  run_cell 6 0.01  1.0 "m001_f10"
  run_cell 6 0.05  1.0 "m005_f10"
  run_cell 6 0.1   1.0 "m01_f10"
) > grid_per_head_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 2 chain PID: ${PID_2}  (follow=0.5,  match in {0.005, 0.01, 0.05, 0.1})"
echo "GPU 6 chain PID: ${PID_6}  (follow=1.0,  match in {0.005, 0.01, 0.05, 0.1})"
echo "Logs:"
echo "  tail -f grid_per_head_gpu2.log"
echo "  tail -f grid_per_head_gpu6.log"
echo "Waiting for both chains to finish..."
wait ${PID_2} ${PID_6}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Grid finished."
