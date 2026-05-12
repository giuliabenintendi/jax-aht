#!/usr/bin/env bash
# Focused grid: hunt persistent escape, not just transient peak.
#
# Background from wandb survey:
#   - JA-only sweeps repeatedly produce transient peaks (best: misunderstood-
#     feather-1321 seed 3 peaked 0.362, regressed to 0.245). Escape is real
#     but doesn't STICK — entropy collapses, PPO destabilizes, policy regresses.
#   - The only PERSISTENT escape in history is lunar-serenity-1301 (COMM ON
#     match=0.05, JA off): seed 1 peaked 0.730, finished 0.718.
#
# So this grid prioritizes hypotheses for keeping escape stable, not for
# finding new transient peaks. All JA cells reuse the prior-best base config
# (m=0.01, f=1.0, a=0.1, per_head=true, gaze=true) and vary ONE knob at a time.
#
# Cells:
#   B1  comm-on replication      — control. Confirms our codebase still escapes
#                                  under the known-good lunar-serenity setup.
#   C1  gaze baseline            — same shaping as prior-best, now with gaze.
#                                  Tests whether gaze alone keeps escape stable.
#   C2  +higher entropy (0.03)   — anti-entropy-collapse via stronger bonus.
#   C3  +higher entropy (0.05)   — even stronger anti-collapse pressure.
#   C4  +lower CLIP_EPS (0.1)    — smaller PPO updates can't move policy as far,
#                                  so a coordination-finding step is less likely
#                                  to be undone by a bad subsequent batch.
#   C5  +SPO loss                — Simple Policy Optimization replaces PPO's
#                                  clipped min with a smooth quadratic penalty;
#                                  reportedly more stable around the optimum.
#   D1  gaze + comm-on hybrid    — comm-on shaping AND gaze AND per-head. If
#                                  comm-only works (B1) and JA-only fails to
#                                  persist (C*), maybe both signals together
#                                  produce a more stable escape.
#
# Shared: per_head=true (except B1 which doesn't use it), NUM_SEEDS=6, 5M steps.
# Estimated wall time per cell: 2-2.5h. 7 cells across 2 GPUs = ~8h overnight.

set -u

NUM_SEEDS=6
TOTAL_STEPS=5e6

# C* cells: vary one PPO-stability knob; otherwise replicate the prior best.
run_ja_cell() {
  local gpu="$1"
  local label="$2"
  local extra_args=("${@:3}")
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} on GPU ${gpu}"
  ./run_gpu.sh "${gpu}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    algorithm.NUM_SEEDS=${NUM_SEEDS} \
    algorithm.TOTAL_TIMESTEPS=${TOTAL_STEPS} \
    algorithm.COMMUNICATION=false \
    task.ENV_KWARGS.match_coef=0.0 \
    task.ENV_KWARGS.stability_coef=0.0 \
    task.ENV_KWARGS.follow_coef=0.0 \
    task.ENV_KWARGS.gaze_mode=true \
    algorithm.JA_CARD_ATTN=true \
    algorithm.JA_PARTNER_FEED_PER_HEAD=true \
    algorithm.JA_ATTN_MATCH_COEF=0.01 \
    algorithm.JA_ATTN_FOLLOW_COEF=1.0 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.1 \
    "${extra_args[@]}"
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit ${rc})"
  return ${rc}
}

run_comm_on_control() {
  local gpu="$1"
  local label="$2"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} on GPU ${gpu} (COMM ON, lunar-serenity replication)"
  ./run_gpu.sh "${gpu}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    algorithm.NUM_SEEDS=${NUM_SEEDS} \
    algorithm.TOTAL_TIMESTEPS=${TOTAL_STEPS} \
    algorithm.COMMUNICATION=true \
    task.ENV_KWARGS.match_coef=0.05 \
    task.ENV_KWARGS.stability_coef=0.0 \
    task.ENV_KWARGS.follow_coef=0.0 \
    algorithm.JA_CARD_ATTN=false \
    algorithm.JA_CARD_METRIC=true \
    algorithm.JA_ATTN_MATCH_COEF=0.0 \
    algorithm.JA_ATTN_FOLLOW_COEF=0.0 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.0
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit ${rc})"
  return ${rc}
}

run_hybrid_d1() {
  local gpu="$1"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] D1_hybrid_comm_ja on GPU ${gpu}"
  ./run_gpu.sh "${gpu}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    algorithm.NUM_SEEDS=${NUM_SEEDS} \
    algorithm.TOTAL_TIMESTEPS=${TOTAL_STEPS} \
    algorithm.COMMUNICATION=true \
    task.ENV_KWARGS.match_coef=0.05 \
    task.ENV_KWARGS.stability_coef=0.0 \
    task.ENV_KWARGS.follow_coef=0.0 \
    task.ENV_KWARGS.gaze_mode=false \
    algorithm.JA_CARD_ATTN=true \
    algorithm.JA_PARTNER_FEED_PER_HEAD=true \
    algorithm.JA_ATTN_MATCH_COEF=0.01 \
    algorithm.JA_ATTN_FOLLOW_COEF=1.0 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.1
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] D1_hybrid_comm_ja (exit ${rc})"
  return ${rc}
}

# GPU 2: control + entropy stability (4 cells)
(
  run_comm_on_control 2 "B1_comm_on_control"
  run_ja_cell 2 "C1_gaze_baseline"
  run_ja_cell 2 "C2_ent_0.03"        algorithm.ENT_COEF=0.03
  run_ja_cell 2 "C3_ent_0.05"        algorithm.ENT_COEF=0.05
) > grid_rich_gpu2.log 2>&1 &
PID_2=$!

# GPU 6: PPO alternatives + hybrid (3 cells)
(
  run_ja_cell 6 "C4_clip_0.1"        algorithm.CLIP_EPS=0.1
  run_ja_cell 6 "C5_spo_loss"        algorithm.POLICY_LOSS_TYPE=spo
  run_hybrid_d1 6
) > grid_rich_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 2 chain PID: ${PID_2}  (B1 control + C1..C3 entropy)"
echo "GPU 6 chain PID: ${PID_6}  (C4 clip + C5 spo + D1 hybrid)"
echo "Logs:"
echo "  tail -f grid_rich_gpu2.log"
echo "  tail -f grid_rich_gpu6.log"
echo "Waiting for both chains to finish..."
wait ${PID_2} ${PID_6}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Grid finished."
