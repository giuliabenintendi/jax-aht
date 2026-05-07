#!/usr/bin/env bash
# Run instrumented self-play diagnostic. Reports empirical SP coord and the
# joint (raw_action_0, raw_action_1) and (gt_pick_0, gt_pick_1) tables.
# Usage:
#   ./diagnose_sp_episode.sh <gpu> [seed_idx]
#   USE_BEST=1 ./diagnose_sp_episode.sh <gpu> [seed_idx]
#   NUM_EPS=500 ./diagnose_sp_episode.sh <gpu> [seed_idx]

GPU="${1:?Usage: ./diagnose_sp_episode.sh <gpu> [seed_idx]}"
SEED_IDX="${2:-3}"
CKPT="${CKPT:-/scratch/benintendi/jax-aht/results/card-game/ja_ippo/default_label/2026-05-04_23-42-20/saved_train_run}"
NUM_EPS="${NUM_EPS:-200}"

EXTRA=()
if [[ "${USE_BEST:-0}" == "1" ]]; then
    EXTRA+=(--use-best)
fi

./run_gpu.sh "${GPU}" evaluation.diagnose_sp_episode \
    --checkpoint "${CKPT}" \
    --seed-idx "${SEED_IDX}" \
    --num-episodes "${NUM_EPS}" \
    "${EXTRA[@]}"
