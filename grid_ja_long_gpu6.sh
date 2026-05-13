#!/usr/bin/env bash
# Long-horizon (1 seed, 15M steps) ablation on GPU 6.
#
# Tests whether stripping the "harvestable without pick-coordination" shaping
# (match, aux) escapes the constant-pick local optimum at a longer training
# horizon than the 5M-step grid.
#
# Cells (all self=0.1):
#   SELF_ONLY:  gaze=0.0, match=0,    aux=0     control — only self, no partner coupling
#   GAZE_ONLY:  gaze=0.5, match=0,    aux=0     minimal proposal: pick must follow partner attention
#   GAZE_AUX:   gaze=0.5, match=0,    aux=0.1   adds aux: does it co-exist with gaze_pick?
#   FULL:       gaze=0.5, match=0.05, aux=0.1   current grid setting at longer horizon
#
# Budget check (match=0.05, self=0.1, gaze_pick=0.5):
#   7 * 0.05 + 0.5 + 0.1 = 0.95 < 1.0 = env_max  (aux is a loss coef, not a reward)
#
# Wall time estimate: ~3h/cell × 4 = ~12h total (overnight).

set -u

NUM_SEEDS=1
TOTAL_STEPS=15e6

run_cell() {
  local gpu="$1"
  local match="$2"
  local self="$3"
  local gaze_pick="$4"
  local aux="$5"
  local label="$6"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} GPU ${gpu} (m=${match} s=${self} g=${gaze_pick} a=${aux})"
  ./run_gpu.sh "${gpu}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    algorithm.NUM_SEEDS=${NUM_SEEDS} \
    algorithm.TOTAL_TIMESTEPS=${TOTAL_STEPS} \
    algorithm.COMMUNICATION=false \
    task.ENV_KWARGS.match_coef=0.0 \
    task.ENV_KWARGS.stability_coef=0.0 \
    task.ENV_KWARGS.follow_coef=0.0 \
    algorithm.JA_CARD_ATTN=true \
    algorithm.JA_PARTNER_FEED_PER_HEAD=true \
    algorithm.JA_ATTN_MATCH_COEF=${match} \
    algorithm.JA_ATTN_SELF_COEF=${self} \
    algorithm.JA_GAZE_PICK_COEF=${gaze_pick} \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=${aux}
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit $?)"
}

(
  run_cell 6 0.00 0.1 0.0 0.00  "SELF_ONLY"
  run_cell 6 0.00 0.1 0.5 0.00  "GAZE_ONLY"
  run_cell 6 0.00 0.1 0.5 0.10  "GAZE_AUX"
  run_cell 6 0.05 0.1 0.5 0.10  "FULL"
) > grid_long_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 6 chain PID: ${PID_6}  (1 seed × 15M, 4 cells)"
echo "Log: tail -f grid_long_gpu6.log"
