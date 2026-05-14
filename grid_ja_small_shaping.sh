#!/usr/bin/env bash
# Small-shaping sweep. Constraint: the shaped *reward* (match + self + gaze)
# must be small relative to the env reward. Every strong run so far violates
# this (shaped ≈ or > env). This sweep keeps self/gaze tiny on purpose and
# leans on aux — which is a supervised LOSS, not a reward, so it does NOT
# count against the shaped-reward budget.
#
# match=0 (per-step, biggest budget risk). self/gaze in {0.01..0.05}: with
# realized firing ~40-50% of max over 8 steps, shaped/episode ≈ 0.1-0.3,
# well under typical env/episode (~0.5). aux is the swept workhorse.
#
# Single GPU (default 2), 1 seed, 15M steps. 6 cells × ~1h = ~6h.
# task=card-game-op-delib-actions (gaze_mode=false).

set -u

GPU="${1:-2}"
SEEDS=1
STEPS=15e6
MATCH=0.0
PER_HEAD=false

run_cell() {
  local self="$1"; local gaze_pick="$2"; local aux="$3"
  local tag="small_match${MATCH}_self${self}_gaze${gaze_pick}_aux${aux}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${tag} GPU ${GPU}"
  MATCH="${MATCH}" SELF="${self}" GAZE_PICK="${gaze_pick}" AUX="${aux}" \
    SEEDS="${SEEDS}" STEPS="${STEPS}" PER_HEAD="${PER_HEAD}" \
    LABEL="${tag}" \
    ./run_ja_delib_actions.sh "${GPU}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${tag} (exit $?)"
}

(
  run_cell 0.02 0.02 0.25   # small shaping, baseline aux
  run_cell 0.02 0.02 0.50   # small shaping, stronger aux
  run_cell 0.02 0.02 1.00   # small shaping, very strong aux
  run_cell 0.01 0.01 0.50   # even smaller shaping
  run_cell 0.03 0.03 0.50   # slightly larger small shaping
  run_cell 0.02 0.05 0.50   # gaze (the coordination signal) a bit larger
) > grid_small_shaping_gpu${GPU}.log 2>&1 &
PID=$!

echo "GPU ${GPU} chain PID: ${PID}  (6 cells, small shaping, 1 seed × 15M each)"
echo "Log: tail -f grid_small_shaping_gpu${GPU}.log"
