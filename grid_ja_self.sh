#!/usr/bin/env bash
# Grid search for the self-consistency reward magnitude. No probe here -
# probe and self-consistency address different problems and should not be
# mixed in the same cells. The probe is a separate single-cell experiment
# (see probe_perfect_feed.sh if needed).
#
# Shared (all cells):
#   - per_head=true, gaze=true, COMMUNICATION=false, NO PROBE
#   - 6 seeds, 5M steps each (~2h/cell at 6 seeds @ 5M)
#
# Sweep:
#   self_coef in {0, 0.1, 0.5, 1.0}, with prior shaping ON or OFF
#
# Total: 8 cells split 4/4 across GPU 2 and GPU 6 -> ~8h overnight.

set -u

NUM_SEEDS=6
TOTAL_STEPS=5e6
PRIOR_M=0.01
PRIOR_F=1.0
PRIOR_A=0.1

run_self_only() {
  local gpu="$1"
  local self="$2"
  local label="$3"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} GPU ${gpu} (self=${self}, no other shaping)"
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
    algorithm.JA_ATTN_MATCH_COEF=0.0 \
    algorithm.JA_ATTN_FOLLOW_COEF=0.0 \
    algorithm.JA_ATTN_SELF_COEF=${self} \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.0
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit $?)"
}

run_self_plus_prior() {
  local gpu="$1"
  local self="$2"
  local label="$3"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} GPU ${gpu} (self=${self} + prior best)"
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
    algorithm.JA_ATTN_MATCH_COEF=${PRIOR_M} \
    algorithm.JA_ATTN_FOLLOW_COEF=${PRIOR_F} \
    algorithm.JA_ATTN_SELF_COEF=${self} \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=${PRIOR_A}
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit $?)"
}

# GPU 2: self-coef sweep alone (no other shaping)
(
  run_self_only 2 0.0  "S1_self_0"
  run_self_only 2 0.1  "S2_self_0.1"
  run_self_only 2 0.5  "S3_self_0.5"
  run_self_only 2 1.0  "S4_self_1.0"
) > grid_self_gpu2.log 2>&1 &
PID_2=$!

# GPU 6: self-coef on top of the prior shaping baseline (m=0.01, f=1.0, a=0.1)
(
  run_self_plus_prior 6 0.0  "C1_prior_only"
  run_self_plus_prior 6 0.1  "C2_prior_self_0.1"
  run_self_plus_prior 6 0.5  "C3_prior_self_0.5"
  run_self_plus_prior 6 1.0  "C4_prior_self_1.0"
) > grid_self_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 2 chain PID: ${PID_2}  (self-only sweep)"
echo "GPU 6 chain PID: ${PID_6}  (self + prior shaping sweep)"
echo "Logs:"
echo "  tail -f grid_self_gpu2.log"
echo "  tail -f grid_self_gpu6.log"
echo "Waiting for both chains to finish..."
wait ${PID_2} ${PID_6}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Grid finished."
