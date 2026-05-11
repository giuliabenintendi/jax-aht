#!/usr/bin/env bash
# Sweep JA attention shaping coefs across 4 magnitudes (10x, 50x, 75x, 100x
# lower than the original 0.1 / 0.5 that Goodhart-farmed the match signal).
#
# Each cell: 6 seeds, 10M steps, gaze-only (no comm channel, no comm shaping,
# OP wrappers on, JA attention shaping the only intrinsic signal).
#
# Cells run in parallel across GPU 5 and GPU 6:
#   GPU 5: 10x then 75x   (two cells, sequential)
#   GPU 6: 50x then 100x  (two cells, sequential)
#
# Total wall time: ~2x the per-cell duration (both chains have 2 cells).
# Logs: sweep_gpu5.log, sweep_gpu6.log.
#
# Usage:
#   bash sweep_ja_coefs.sh        # foreground; tail logs in another terminal
#   nohup bash sweep_ja_coefs.sh & # background; survives disconnect

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
    algorithm.JA_ATTN_MATCH_COEF=${match} \
    algorithm.JA_ATTN_FOLLOW_COEF=${follow}
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] finished ${label} (exit ${rc})"
  return ${rc}
}

# GPU 5 chain: 10x lower, then 75x lower
(
  run_cell 5 0.01    0.05    "10x_lower"
  run_cell 5 0.00133 0.00667 "75x_lower"
) > sweep_gpu5.log 2>&1 &
PID_5=$!

# GPU 6 chain: 50x lower, then 100x lower
(
  run_cell 6 0.002 0.01  "50x_lower"
  run_cell 6 0.001 0.005 "100x_lower"
) > sweep_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 5 chain PID: ${PID_5} (cells: 10x_lower then 75x_lower)"
echo "GPU 6 chain PID: ${PID_6} (cells: 50x_lower then 100x_lower)"
echo "Tail logs:"
echo "  tail -f sweep_gpu5.log"
echo "  tail -f sweep_gpu6.log"
echo "Waiting for both chains to finish..."
wait ${PID_5} ${PID_6}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] All cells finished."
