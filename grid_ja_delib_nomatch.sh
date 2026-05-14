#!/usr/bin/env bash
# match=0 sweep. Diagnosis from the best runs (c4z5j3uj, 65oi48mi): `match`
# saturates at ~0.2M steps (two diffuse attentions agree trivially), then sits
# as a flat offset — it provides no learning gradient. `self` and `gaze` are
# the env-aligned terms (corr +0.5..+0.77, saturate with env). So drop match
# and re-tune self/gaze/aux around the known-good c4z5j3uj point.
#
# Single GPU (default 1), 1 seed, 15M steps. 6 cells × ~1h = ~6h.
#
# Cell 1 is the direct "drop match" test: c4z5j3uj's coefs with match=0.

set -u

GPU="${1:-1}"
SEEDS=1
STEPS=15e6
PER_HEAD=false
MATCH=0.0

run_cell() {
  local self="$1"; local gaze_pick="$2"; local aux="$3"
  local tag="match${MATCH}_self${self}_gaze${gaze_pick}_aux${aux}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${tag} GPU ${GPU}"
  MATCH="${MATCH}" SELF="${self}" GAZE_PICK="${gaze_pick}" AUX="${aux}" \
    SEEDS="${SEEDS}" STEPS="${STEPS}" PER_HEAD="${PER_HEAD}" \
    ./run_ja_delib_actions.sh "${GPU}"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${tag} (exit $?)"
}

(
  run_cell 0.10 0.05 0.25   # drop-match baseline: c4z5j3uj coefs, match off
  run_cell 0.15 0.05 0.25   # more self
  run_cell 0.10 0.10 0.25   # more gaze
  run_cell 0.15 0.10 0.25   # more self + gaze
  run_cell 0.20 0.10 0.25   # even more self
  run_cell 0.15 0.10 0.40   # more self + gaze + stronger aux
) > grid_nomatch_gpu${GPU}.log 2>&1 &
PID=$!

echo "GPU ${GPU} chain PID: ${PID}  (6 cells, match=0, 1 seed × 15M each)"
echo "Log: tail -f grid_nomatch_gpu${GPU}.log"
