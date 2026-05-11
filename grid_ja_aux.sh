#!/usr/bin/env bash
# 3x2 grid sweep over (follow_coef, aux_coef) for gaze-only JA + aux + partner-feed.
#
# Shared config (all cells):
#   - gaze-only (no comm channel, no comm shaping)
#   - JA_CARD_ATTN=true → partner's canonical card attention as obs input
#   - JA_ATTN_MATCH_COEF=0.0 (skip — Goodhart-prone in prior sweep)
#   - OP wrappers on
#   - 6 seeds, 5M steps per cell
#
# Grid:
#   follow ∈ {0.05, 0.1, 0.5}
#   aux    ∈ {0.0, 0.1}
#
# Sequential on GPU 6. ~30 min per cell × 6 = ~3 hours total.
# Logs:  grid_ja_aux.log  (one log, cells separated by [start]/[finish] timestamps)

set -u

NUM_SEEDS=6
TOTAL_STEPS=5e6
GPU=6

run_cell() {
  local follow="$1"
  local aux="$2"
  local label="$3"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} (follow=${follow}, aux=${aux})"
  ./run_gpu.sh "${GPU}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    algorithm.NUM_SEEDS=${NUM_SEEDS} \
    algorithm.TOTAL_TIMESTEPS=${TOTAL_STEPS} \
    algorithm.COMMUNICATION=false \
    task.ENV_KWARGS.match_coef=0.0 \
    task.ENV_KWARGS.stability_coef=0.0 \
    task.ENV_KWARGS.follow_coef=0.0 \
    algorithm.JA_CARD_ATTN=true \
    algorithm.JA_ATTN_MATCH_COEF=0.0 \
    algorithm.JA_ATTN_FOLLOW_COEF=${follow} \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=${aux}
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit ${rc})"
  return ${rc}
}

(
  run_cell 0.05 0.0 "follow005_aux000"
  run_cell 0.05 0.1 "follow005_aux010"
  run_cell 0.1  0.0 "follow010_aux000"
  run_cell 0.1  0.1 "follow010_aux010"
  run_cell 0.5  0.0 "follow050_aux000"
  run_cell 0.5  0.1 "follow050_aux010"
) > grid_ja_aux.log 2>&1 &
PID=$!

echo "Grid PID: ${PID}"
echo "GPU: ${GPU}"
echo "Cells (sequential):"
echo "  1) follow=0.05  aux=0.0"
echo "  2) follow=0.05  aux=0.1"
echo "  3) follow=0.10  aux=0.0"
echo "  4) follow=0.10  aux=0.1"
echo "  5) follow=0.50  aux=0.0"
echo "  6) follow=0.50  aux=0.1"
echo "Log: tail -f grid_ja_aux.log"
echo "Waiting..."
wait ${PID}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Grid finished."
