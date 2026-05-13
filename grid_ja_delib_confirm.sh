#!/usr/bin/env bash
# Multi-seed confirmation of the best stable config (c4z5j3uj).
#
# Single cell on GPU ${1:-1}:
#   match=0.05, self=0.10, gaze=0.05, aux=0.25
#
# Defaults (override via env vars):
#   SEEDS=4, STEPS=20e6  → ~5.3h on one GPU (15M seed-steps/hour observed).
#
# To fit a tighter 6h budget with 6 seeds, set SEEDS=6 STEPS=15e6.
# To go all-in on 6 seeds × 20M (~8h), accept overrun and set SEEDS=6 STEPS=20e6.

set -u

GPU="${1:-1}"
SEEDS="${SEEDS:-4}"
STEPS="${STEPS:-20e6}"
MATCH=0.05
SELF=0.10
GAZE_PICK=0.05
AUX=0.25
PER_HEAD=false

TAG="match${MATCH}_self${SELF}_gaze${GAZE_PICK}_aux${AUX}_${SEEDS}seeds"
echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${TAG} GPU ${GPU} seeds=${SEEDS} steps=${STEPS}"

MATCH="${MATCH}" SELF="${SELF}" GAZE_PICK="${GAZE_PICK}" AUX="${AUX}" \
  SEEDS="${SEEDS}" STEPS="${STEPS}" PER_HEAD="${PER_HEAD}" \
  LABEL="${TAG}" \
  ./run_ja_delib_actions.sh "${GPU}" > grid_confirm_gpu${GPU}.log 2>&1 &
PID=$!

echo "Chain PID: ${PID}"
echo "Log: tail -f grid_confirm_gpu${GPU}.log"
