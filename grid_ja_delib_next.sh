#!/usr/bin/env bash
# Follow-up to grid_ja_delib_15M.sh based on the first sweep's results.
# Each cell = 15M steps. Top wall time per cell ~1h (1 seed) up to ~3h (3 seeds).
# 3 GPUs × 2 cells = 6 cells.
#
# GPU 1 — multi-seed confirmation of the two best configs from the previous sweep
#   (peak 0.75 single-seed → does it hold across 3 seeds?):
#     match=0.05, self=0.10, gaze=0.05, aux=0.10   (9rjvkdka — best final)
#     match=0.05, self=0.05, gaze=0.05, aux=0.10   (cyi6mt1q — best peak)
#
# GPU 2 — "fair" experiment: drop self (the most artificial / self-referential reward).
#   Tests whether coordination still emerges from partner-conditioned signals
#   (aux + gaze_pick) alone, with no self-imposed attention/action coupling.
#     match=0.05, self=0,    gaze=0.05, aux=0.10
#     match=0,    self=0,    gaze=0.05, aux=0.10   (only partner-conditional signals)
#
# GPU 3 — stronger aux to attack the instability (aux_nll drifting back up under
#   PPO over-updates). Higher aux pins attention harder against actor-side drift.
#     match=0.05, self=0.05, gaze=0.05, aux=0.25
#     match=0.05, self=0.10, gaze=0.05, aux=0.25

set -u

STEPS=15e6
PER_HEAD=false

run_cell() {
  local gpu="$1"; local match="$2"; local self="$3"; local gaze_pick="$4"; local aux="$5"; local seeds="$6"
  local tag="match${match}_self${self}_gaze${gaze_pick}_aux${aux}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${tag} GPU ${gpu} seeds=${seeds}"
  MATCH="${match}" SELF="${self}" GAZE_PICK="${gaze_pick}" AUX="${aux}" \
    SEEDS="${seeds}" STEPS="${STEPS}" PER_HEAD="${PER_HEAD}" \
    ./run_ja_delib_actions.sh "${gpu}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${tag} (exit $?)"
}

# GPU 1 — multi-seed (3 seeds) confirmation of best configs
(
  run_cell 1 0.05 0.10 0.05 0.10  3
  run_cell 1 0.05 0.05 0.05 0.10  3
) > grid_next_gpu1.log 2>&1 &
PID_1=$!

# GPU 2 — drop self (fairness test, 1 seed each)
(
  run_cell 2 0.05 0.00 0.05 0.10  1
  run_cell 2 0.00 0.00 0.05 0.10  1
) > grid_next_gpu2.log 2>&1 &
PID_2=$!

# GPU 3 — stronger aux (instability test, 1 seed each)
(
  run_cell 3 0.05 0.05 0.05 0.25  1
  run_cell 3 0.05 0.10 0.05 0.25  1
) > grid_next_gpu3.log 2>&1 &
PID_3=$!

echo "GPU 1 chain PID: ${PID_1}  (multi-seed confirmation)"
echo "GPU 2 chain PID: ${PID_2}  (drop-self ablation)"
echo "GPU 3 chain PID: ${PID_3}  (stronger aux)"
echo "Logs:"
echo "  tail -f grid_next_gpu1.log"
echo "  tail -f grid_next_gpu2.log"
echo "  tail -f grid_next_gpu3.log"
