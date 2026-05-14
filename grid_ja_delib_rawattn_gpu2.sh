#!/usr/bin/env bash
# Complement to grid_ja_delib_rawattn.sh (GPU 1). That sweep found 4× scale
# (m=0.20, s=0.40, g=0.20) the best so far (peak 0.55); 1-2x were dead.
# This GPU-2 sweep fills the gaps:
#   - does scale keep helping past 4× (5×, 6×)?
#   - is match needed at the 4× anchor (GPU 1 only tested no-match at 3×)?
#   - does the 4× point want stronger aux?
#
# Single GPU (default 2), 1 seed, 15M steps. 4 cells × ~1h = ~4h wall.

set -u

GPU="${1:-2}"
SEEDS=1
STEPS=15e6
PER_HEAD=false

run_cell() {
  local match="$1"; local self="$2"; local gaze_pick="$3"; local aux="$4"
  local tag="match${match}_self${self}_gaze${gaze_pick}_aux${aux}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${tag} GPU ${GPU}"
  MATCH="${match}" SELF="${self}" GAZE_PICK="${gaze_pick}" AUX="${aux}" \
    SEEDS="${SEEDS}" STEPS="${STEPS}" PER_HEAD="${PER_HEAD}" \
    ./run_ja_delib_actions.sh "${GPU}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${tag} (exit $?)"
}

(
  run_cell 0.25 0.50 0.25 0.25   # 5× scale
  run_cell 0.30 0.60 0.30 0.25   # 6× scale
  run_cell 0.00 0.40 0.20 0.25   # 4× anchor, no match
  run_cell 0.20 0.40 0.20 0.40   # 4× anchor, stronger aux
) > grid_rawattn_gpu${GPU}.log 2>&1 &
PID=$!

echo "GPU ${GPU} chain PID: ${PID}  (4 cells, 1 seed × 15M each)"
echo "Log: tail -f grid_rawattn_gpu${GPU}.log"
