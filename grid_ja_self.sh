#!/usr/bin/env bash
# Sweep over JA_ATTN_SELF_COEF, with and without the prior shaping baseline.
# Aux is fixed at 0.1 (the target-frame fix means it is now a learnable
# signal, not noise). Match and follow are either off (S row) or at the
# prior best (m=0.01, f=1.0) (C row).
#
# Shared: per_head=true, gaze=true, COMMUNICATION=false, NO PROBE.
#         6 seeds, 5M steps each (~2h/cell).
#
# Sweep: self in {0, 0.1, 0.5, 1.0, 2.0}, prior shaping on/off.
# Total: 10 cells. 5 cells per GPU. ~10h overnight.

set -u

NUM_SEEDS=6
TOTAL_STEPS=5e6
PRIOR_M=0.01
PRIOR_F=1.0
AUX_FIXED=0.1

run_self_only() {
  local gpu="$1"
  local self="$2"
  local label="$3"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} GPU ${gpu} (self=${self}, no match/follow, aux=${AUX_FIXED})"
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
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=${AUX_FIXED}
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit $?)"
}

run_self_plus_prior() {
  local gpu="$1"
  local self="$2"
  local label="$3"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} GPU ${gpu} (self=${self} + prior m=${PRIOR_M} f=${PRIOR_F} aux=${AUX_FIXED})"
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
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=${AUX_FIXED}
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit $?)"
}

# GPU 2: self-coef alone (no match, no follow), 5 cells
(
  run_self_only 2 0.0  "S1_self_0"
  run_self_only 2 0.1  "S2_self_0.1"
  run_self_only 2 0.5  "S3_self_0.5"
  run_self_only 2 1.0  "S4_self_1.0"
  run_self_only 2 2.0  "S5_self_2.0"
) > grid_self_gpu2.log 2>&1 &
PID_2=$!

# GPU 6: self-coef on top of prior shaping (m=0.01 f=1.0 a=0.1), 5 cells
(
  run_self_plus_prior 6 0.0  "C1_prior_self_0"
  run_self_plus_prior 6 0.1  "C2_prior_self_0.1"
  run_self_plus_prior 6 0.5  "C3_prior_self_0.5"
  run_self_plus_prior 6 1.0  "C4_prior_self_1.0"
  run_self_plus_prior 6 2.0  "C5_prior_self_2.0"
) > grid_self_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 2 chain PID: ${PID_2}  (self-only sweep: S1..S5)"
echo "GPU 6 chain PID: ${PID_6}  (self + prior sweep: C1..C5)"
echo "Logs:"
echo "  tail -f grid_self_gpu2.log"
echo "  tail -f grid_self_gpu6.log"
echo "Waiting for both chains to finish..."
wait ${PID_2} ${PID_6}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Grid finished."
