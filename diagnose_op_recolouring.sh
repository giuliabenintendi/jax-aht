#!/usr/bin/env bash
# Diagnose whether OP per-agent recolourings are actually independent at eval.
# Usage:
#   ./diagnose_op_recolouring.sh <gpu>
#   CKPT=/path/to/saved_train_run ./diagnose_op_recolouring.sh <gpu>

GPU="${1:?Usage: ./diagnose_op_recolouring.sh <gpu>}"
CKPT="${CKPT:-/scratch/benintendi/jax-aht/results/card-game/ja_ippo/default_label/2026-05-04_23-42-20/saved_train_run}"
NUM_RESETS="${NUM_RESETS:-200}"

./run_gpu.sh "${GPU}" evaluation.diagnose_op_recolouring \
    --checkpoint "${CKPT}" \
    --num-resets "${NUM_RESETS}"
