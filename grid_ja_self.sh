#!/usr/bin/env bash
# Sweep over JA_ATTN_SELF_COEF, with and without the prior shaping baseline.
# Aux is fixed at 0.1 (the target-frame fix means it is now a learnable
# signal, not noise). Match and follow are either off (S row) or at the
# prior best (m=0.01, f=1.0) (C row).
#
# Shared: per_head=true, gaze=true, COMMUNICATION=false, NO PROBE.
#         3 seeds, 5M steps each (~1h/cell).
#
# Sweep: self in {0.05, 0.1, 0.15}, prior shaping on/off.
# Max per-agent shaped reward without self = match (7*0.05) + follow (0.5) = 0.85.
# Constrained to keep total shaped < env_max (1.0), so self < 0.15.
# Total: 6 cells. Runs sequentially on GPU 6. ~6h.

set -u

NUM_SEEDS=3
TOTAL_STEPS=5e6
PRIOR_M=0.05
PRIOR_F=0.5
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

# All 6 cells sequentially on GPU 6.
(
  run_self_only       6 0.05  "S1_self_0.05"
  run_self_only       6 0.10  "S2_self_0.10"
  run_self_only       6 0.15  "S3_self_0.15"
  run_self_plus_prior 6 0.05  "C1_prior_self_0.05"
  run_self_plus_prior 6 0.10  "C2_prior_self_0.10"
  run_self_plus_prior 6 0.15  "C3_prior_self_0.15"
) > grid_self_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 6 chain PID: ${PID_6}  (S1..S3 + C1..C3 sequentially)"
echo "Log: tail -f grid_self_gpu6.log"
echo "Waiting for chain to finish..."
wait ${PID_6}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Grid finished."
