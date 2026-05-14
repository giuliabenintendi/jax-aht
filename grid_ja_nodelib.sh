#!/usr/bin/env bash
# Sweep over the no-deliberation-action config (task=card-game-op-nodelib,
# gaze_mode=true). Deliberation steps emit no action (forced noop slot); the
# decision-step pick is the only real action. `match` fires per deliberation
# step; `self`/`gaze_pick` fire only at the decision step.
#
# Goal: find whether *any* coefficient combination escapes chance in nodelib
# mode (the open question — this design caused entropy collapse before; aux now
# pins the attention head every step, which is the new factor).
#
# Sweep axes: gaze_pick (the decision-step coordination signal) × aux.
# match=0 (per-step, biggest shaped-budget risk — start off);
# self=0.05 fixed (small decision-step anchor).
# With match=0, max shaped/episode ≈ self + gaze ≤ 0.05 + 0.5 = 0.55 < env(1.0).
#
# Single GPU (default 2), 1 seed, 15M steps. 6 cells × ~1h = ~6h.

set -u

GPU="${1:-2}"
SEEDS=1
STEPS=15e6
MATCH=0.0
SELF=0.05
PER_HEAD=false

run_cell() {
  local gaze_pick="$1"; local aux="$2"
  local tag="nodelib_match${MATCH}_self${SELF}_gaze${gaze_pick}_aux${aux}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${tag} GPU ${GPU}"
  MATCH="${MATCH}" SELF="${SELF}" GAZE_PICK="${gaze_pick}" AUX="${aux}" \
    SEEDS="${SEEDS}" STEPS="${STEPS}" PER_HEAD="${PER_HEAD}" \
    LABEL="${tag}" \
    ./run_ja_nodelib.sh "${GPU}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${tag} (exit $?)"
}

(
  run_cell 0.10 0.10
  run_cell 0.10 0.25
  run_cell 0.30 0.10
  run_cell 0.30 0.25
  run_cell 0.50 0.10
  run_cell 0.50 0.25
) > grid_nodelib_gpu${GPU}.log 2>&1 &
PID=$!

echo "GPU ${GPU} chain PID: ${PID}  (6 cells, nodelib mode, 1 seed × 15M each)"
echo "Log: tail -f grid_nodelib_gpu${GPU}.log"
