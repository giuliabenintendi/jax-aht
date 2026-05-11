#!/usr/bin/env bash
# Grid sweep with per-head partner-feed (20-dim instead of 5-dim).
#
# Shared config (all cells):
#   - gaze-only (no comm, no comm shaping)
#   - JA_CARD_ATTN=true → partner-feed input pathway on
#   - JA_PARTNER_FEED_PER_HEAD=true → 20-dim per-head feed (NEW)
#   - JA_ATTN_MATCH_COEF=0.0  (skip — Goodhart-prone)
#   - OP wrappers on
#   - 6 seeds, 10M steps per cell  (~1 hour/cell)
#
# Grid (8 cells):
#   follow ∈ {0.1, 0.5, 1.0, 2.0}
#   aux    ∈ {0.0, 0.1}
#
# Parallel across GPU 2 and GPU 6:
#   GPU 2 handles 4 cells sequentially
#   GPU 6 handles 4 cells sequentially
# Total wall time: ~4 hours per GPU = 4 hours overall.

set -u

NUM_SEEDS=6
TOTAL_STEPS=10e6

run_cell() {
  local gpu="$1"
  local follow="$2"
  local aux="$3"
  local label="$4"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} on GPU ${gpu} (follow=${follow}, aux=${aux})"
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
    algorithm.JA_ATTN_MATCH_COEF=0.0 \
    algorithm.JA_ATTN_FOLLOW_COEF=${follow} \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=${aux}
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit ${rc})"
  return ${rc}
}

# GPU 2: cells 1-4 (follow=0.1 and 0.5, both aux values)
(
  run_cell 2 0.1  0.0  "f01_aux0"
  run_cell 2 0.1  0.1  "f01_aux01"
  run_cell 2 0.5  0.0  "f05_aux0"
  run_cell 2 0.5  0.1  "f05_aux01"
) > grid_per_head_gpu2.log 2>&1 &
PID_2=$!

# GPU 6: cells 5-8 (follow=1.0 and 2.0, both aux values)
(
  run_cell 6 1.0  0.0  "f10_aux0"
  run_cell 6 1.0  0.1  "f10_aux01"
  run_cell 6 2.0  0.0  "f20_aux0"
  run_cell 6 2.0  0.1  "f20_aux01"
) > grid_per_head_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 2 chain PID: ${PID_2}  (follow=0.1, 0.5  x  aux=0, 0.1)"
echo "GPU 6 chain PID: ${PID_6}  (follow=1.0, 2.0  x  aux=0, 0.1)"
echo "Logs:"
echo "  tail -f grid_per_head_gpu2.log"
echo "  tail -f grid_per_head_gpu6.log"
echo "Waiting for both chains to finish..."
wait ${PID_2} ${PID_6}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Grid finished."
