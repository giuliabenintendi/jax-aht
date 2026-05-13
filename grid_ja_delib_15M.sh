#!/usr/bin/env bash
# 15M, 1-seed sweep over the supervisor's new per-step JA shaping rewards.
# Goal: find which combination of (match, self, gaze_pick, aux) escapes chance.
#
# Each cell: ~3h. 3 GPUs × 3 cells sequential = ~9h wall.
# Run names are LABEL=match{x}_self{x}_gaze{x}_aux{x}.
#
# GPU 1 — no match, no aux (minimal coupling):
#   m=0,    s=0.05, g=0
#   m=0,    s=0.05, g=0.05
#   m=0,    s=0.05, g=0.10
#
# GPU 2 — with match, no aux (supervisor flavour):
#   m=0.05, s=0.05, g=0
#   m=0.05, s=0.05, g=0.05
#   m=0.05, s=0.10, g=0           (supervisor's exact default)
#
# GPU 3 — aux=0.1 on top:
#   m=0,    s=0.05, g=0.05
#   m=0.05, s=0.05, g=0.05
#   m=0.05, s=0.10, g=0.05

set -u

STEPS=15e6
SEEDS=1
PER_HEAD=false

run_cell() {
  local gpu="$1"; local match="$2"; local self="$3"; local gaze_pick="$4"; local aux="$5"
  local tag="match${match}_self${self}_gaze${gaze_pick}_aux${aux}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${tag} GPU ${gpu}"
  MATCH="${match}" SELF="${self}" GAZE_PICK="${gaze_pick}" AUX="${aux}" \
    SEEDS="${SEEDS}" STEPS="${STEPS}" PER_HEAD="${PER_HEAD}" \
    ./run_ja_delib_actions.sh "${gpu}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${tag} (exit $?)"
}

# GPU 1
(
  run_cell 1 0.00 0.05 0.00 0.00
  run_cell 1 0.00 0.05 0.05 0.00
  run_cell 1 0.00 0.05 0.10 0.00
) > grid_delib_gpu1.log 2>&1 &
PID_1=$!

# GPU 2
(
  run_cell 2 0.05 0.05 0.00 0.00
  run_cell 2 0.05 0.05 0.05 0.00
  run_cell 2 0.05 0.10 0.00 0.00
) > grid_delib_gpu2.log 2>&1 &
PID_2=$!

# GPU 3
(
  run_cell 3 0.00 0.05 0.05 0.10
  run_cell 3 0.05 0.05 0.05 0.10
  run_cell 3 0.05 0.10 0.05 0.10
) > grid_delib_gpu3.log 2>&1 &
PID_3=$!

echo "GPU 1 chain PID: ${PID_1}"
echo "GPU 2 chain PID: ${PID_2}"
echo "GPU 3 chain PID: ${PID_3}"
echo "Logs:"
echo "  tail -f grid_delib_gpu1.log"
echo "  tail -f grid_delib_gpu2.log"
echo "  tail -f grid_delib_gpu3.log"
