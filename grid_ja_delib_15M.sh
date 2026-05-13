#!/usr/bin/env bash
# 15M, 1-seed sweep over the supervisor's new per-step JA shaping rewards.
# Goal: find which combination of (match, self, gaze_pick, aux) escapes chance.
#
# Each cell: ~3h. 3 GPUs × 3 cells sequential = ~9h wall.
#
# GPU 1 — no match, no aux (minimal coupling):
#   A: m=0,    s=0.05, g=0,    a=0      self only (per-agent anchor)
#   B: m=0,    s=0.05, g=0.05, a=0      self + gaze_pick
#   C: m=0,    s=0.05, g=0.10, a=0      self + stronger gaze_pick
#
# GPU 2 — with match, no aux (supervisor flavour):
#   D: m=0.05, s=0.05, g=0,    a=0      match + self
#   E: m=0.05, s=0.05, g=0.05, a=0      full shaping (lower magnitude)
#   F: m=0.05, s=0.10, g=0,    a=0      supervisor's exact default
#
# GPU 3 — aux on top:
#   G: m=0,    s=0.05, g=0.05, a=0.1    minimal + aux
#   H: m=0.05, s=0.05, g=0.05, a=0.1    full + aux
#   I: m=0.05, s=0.10, g=0.05, a=0.1    supervisor's + gaze + aux

set -u

STEPS=15e6
SEEDS=1
PER_HEAD=false

run_cell() {
  local gpu="$1"; local match="$2"; local self="$3"; local gaze_pick="$4"; local aux="$5"; local label="$6"
  local run_label="${label}_m${match}_s${self}_g${gaze_pick}_a${aux}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${run_label} GPU ${gpu}"
  MATCH="${match}" SELF="${self}" GAZE_PICK="${gaze_pick}" AUX="${aux}" \
    SEEDS="${SEEDS}" STEPS="${STEPS}" PER_HEAD="${PER_HEAD}" \
    LABEL="${run_label}" \
    ./run_ja_delib_actions.sh "${gpu}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${run_label} (exit $?)"
}

# GPU 1
(
  run_cell 1 0.00 0.05 0.00 0.00  "A_self_only"
  run_cell 1 0.00 0.05 0.05 0.00  "B_self_gaze"
  run_cell 1 0.00 0.05 0.10 0.00  "C_self_strongergaze"
) > grid_delib_gpu1.log 2>&1 &
PID_1=$!

# GPU 2
(
  run_cell 2 0.05 0.05 0.00 0.00  "D_match_self"
  run_cell 2 0.05 0.05 0.05 0.00  "E_full_low"
  run_cell 2 0.05 0.10 0.00 0.00  "F_supervisor_default"
) > grid_delib_gpu2.log 2>&1 &
PID_2=$!

# GPU 3
(
  run_cell 3 0.00 0.05 0.05 0.10  "G_minimal_aux"
  run_cell 3 0.05 0.05 0.05 0.10  "H_full_aux"
  run_cell 3 0.05 0.10 0.05 0.10  "I_supervisor_plus_aux"
) > grid_delib_gpu3.log 2>&1 &
PID_3=$!

echo "GPU 1 chain PID: ${PID_1}  (A_self_only -> B_self_gaze -> C_self_strongergaze)"
echo "GPU 2 chain PID: ${PID_2}  (D_match_self -> E_full_low -> F_supervisor_default)"
echo "GPU 3 chain PID: ${PID_3}  (G_minimal_aux -> H_full_aux -> I_supervisor_plus_aux)"
echo "Logs:"
echo "  tail -f grid_delib_gpu1.log"
echo "  tail -f grid_delib_gpu2.log"
echo "  tail -f grid_delib_gpu3.log"
