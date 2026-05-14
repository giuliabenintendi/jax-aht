#!/usr/bin/env bash
# Post-reward-fix sweep (commit 7880e10): match/self/gaze_pick now key off the
# raw on-card attention mass instead of the normalized distribution, which
# makes them ~2-5x weaker in effective magnitude (phys = q_phys * m, m < 1).
#
# This sweep re-calibrates the coefficient scale. It scales the previous best
# ratios (c4z5j3uj: m=0.05, s=0.10, g=0.05) up by 1x..4x to find the new
# operating point. aux is UNAFFECTED by the fix (uses argmax + m-weight, not
# the reward path), so it stays at 0.25 except in the last cell.
#
# Single GPU (default 1), 1 seed, 15M steps each. 6 cells × ~1h = ~6h wall.

set -u

GPU="${1:-1}"
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
  # scale sweep on (match, self, gaze) — old best ratios × {1, 2, 3, 4}
  run_cell 0.05 0.10 0.05 0.25   # 1x — old values, expected too weak post-fix
  run_cell 0.10 0.20 0.10 0.25   # 2x
  run_cell 0.15 0.30 0.15 0.25   # 3x
  run_cell 0.20 0.40 0.20 0.25   # 4x
  # at 3x scale: does mass-gated match still help, and does stronger aux help?
  run_cell 0.00 0.30 0.15 0.25   # 3x, no match
  run_cell 0.15 0.30 0.15 0.40   # 3x, stronger aux
) > grid_rawattn_gpu${GPU}.log 2>&1 &
PID=$!

echo "GPU ${GPU} chain PID: ${PID}  (6 cells, 1 seed × 15M each)"
echo "Log: tail -f grid_rawattn_gpu${GPU}.log"
