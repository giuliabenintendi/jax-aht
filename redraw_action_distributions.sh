#!/usr/bin/env bash
# Redraw action distributions with best_params, both OP-on and OP-off, for the
# baseline run. Outputs go to two parallel directories so they can be compared.
#
# Usage:
#   ./redraw_action_distributions.sh <gpu>
#   CKPT=/some/saved_train_run NUM_EPS=500 ./redraw_action_distributions.sh <gpu>

GPU="${1:?Usage: ./redraw_action_distributions.sh <gpu>}"
CKPT="${CKPT:-/scratch/benintendi/jax-aht/results/card-game/ja_ippo/default_label/2026-05-04_23-42-20/saved_train_run}"
NUM_EPS="${NUM_EPS:-500}"

# Build a tag from the run dir so the two output dirs sit next to each other.
RUN_DIR="$(dirname "${CKPT}")"
LABEL_TS="$(basename "$(dirname "${RUN_DIR}")")_$(basename "${RUN_DIR}")"
PLOT_ROOT="/scratch/benintendi/jax-aht/plots/card_game"

OUT_OP_ON="${PLOT_ROOT}/${LABEL_TS}_BEST_OP_ON"
OUT_OP_OFF="${PLOT_ROOT}/${LABEL_TS}_BEST_OP_OFF"

echo "[$(date +%H:%M)] OP-ON  best_params action distributions -> ${OUT_OP_ON}"
mkdir -p "${OUT_OP_ON}"
./run_gpu.sh "${GPU}" evaluation.action_distributions \
    --checkpoint "${CKPT}" \
    --use-best \
    --all-seeds \
    --num-episodes "${NUM_EPS}" \
    --output-dir "${OUT_OP_ON}"

echo "[$(date +%H:%M)] OP-OFF best_params action distributions -> ${OUT_OP_OFF}"
mkdir -p "${OUT_OP_OFF}"
./run_gpu.sh "${GPU}" evaluation.action_distributions \
    --checkpoint "${CKPT}" \
    --use-best \
    --drop-op \
    --all-seeds \
    --num-episodes "${NUM_EPS}" \
    --output-dir "${OUT_OP_OFF}"

echo "[$(date +%H:%M)] done"
