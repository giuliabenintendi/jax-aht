#!/usr/bin/env bash
# 3-GPU sweep of JA_GAZE_PICK_COEF, with three different shaping baselines.
# Each GPU runs 4 cells sequentially, varying gaze_pick in {0, 0.1, 0.3, 0.5}.
#
# Shared (all cells):
#   - per_head=true, gaze_mode=true (task default), COMMUNICATION=false
#   - aux=0.1 (now reads from attention pool directly with causal target)
#   - 3 seeds, 5M steps each (~1h/cell)
#
# GPU 1 (G row): gaze_pick alone (no match, no self).
#   Tests: does gaze_pick alone escape? At what magnitude?
#
# GPU 2 (GS row): gaze_pick + self.
#   Tests: does the self anchor help on top of gaze_pick?
#
# GPU 3 (GMS row): gaze_pick + match + self (full shaping).
#   Tests: does match still contribute when gaze_pick is the main coordination signal?
#
# Budget check (match=0.05, self=0.1, gaze_pick=0.5):
#   7 * 0.05 + 0.5 + 0.1 = 0.95 < 1.0 = env_max
# So all cells respect shaped < env.

set -u

NUM_SEEDS=3
TOTAL_STEPS=5e6
AUX_FIXED=0.1

run_cell() {
  local gpu="$1"
  local match="$2"
  local self="$3"
  local gaze_pick="$4"
  local label="$5"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} GPU ${gpu} (m=${match} s=${self} g=${gaze_pick})"
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
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=${AUX_FIXED}
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit $?)"
}

# GPU 1: gaze_pick alone (m=0, s=0)
(
  run_cell 1 0.00 0.0 0.0  "G1_gp_0"
  run_cell 1 0.00 0.0 0.1  "G2_gp_0.1"
  run_cell 1 0.00 0.0 0.3  "G3_gp_0.3"
  run_cell 1 0.00 0.0 0.5  "G4_gp_0.5"
) > grid_gp_gpu1.log 2>&1 &
PID_1=$!

# GPU 2: gaze_pick + self (m=0, s=0.1)
(
  run_cell 2 0.00 0.1 0.0  "GS1_gp_0"
  run_cell 2 0.00 0.1 0.1  "GS2_gp_0.1"
  run_cell 2 0.00 0.1 0.3  "GS3_gp_0.3"
  run_cell 2 0.00 0.1 0.5  "GS4_gp_0.5"
) > grid_gp_gpu2.log 2>&1 &
PID_2=$!

# GPU 3: gaze_pick + match + self (m=0.05, s=0.1)
(
  run_cell 3 0.05 0.1 0.0  "GMS1_gp_0"
  run_cell 3 0.05 0.1 0.1  "GMS2_gp_0.1"
  run_cell 3 0.05 0.1 0.3  "GMS3_gp_0.3"
  run_cell 3 0.05 0.1 0.5  "GMS4_gp_0.5"
) > grid_gp_gpu3.log 2>&1 &
PID_3=$!

echo "GPU 1 chain PID: ${PID_1}  (G row: gaze_pick alone)"
echo "GPU 2 chain PID: ${PID_2}  (GS row: gaze_pick + self=0.1)"
echo "GPU 3 chain PID: ${PID_3}  (GMS row: gaze_pick + match=0.05 + self=0.1)"
echo "Logs:"
echo "  tail -f grid_gp_gpu1.log"
echo "  tail -f grid_gp_gpu2.log"
echo "  tail -f grid_gp_gpu3.log"
echo "Waiting for all chains..."
wait ${PID_1} ${PID_2} ${PID_3}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Grid finished."
