#!/usr/bin/env bash
# Rich grid sweep across (match, follow, aux) — hunt for persistent escape.
#
# Context: wandb survey of prior runs shows that JA-only comm-off sweeps
# produce TRANSIENT peaks (best so far: misunderstood-feather-1321 seed 3
# peaked 0.362 then regressed to 0.245). The only PERSISTENT escape in
# history is lunar-serenity-1301 (COMM ON + env match=0.05): seed 1
# peaked 0.730, finished 0.718. So this grid does two things:
#   (a) re-explore JA-only under the new gaze_mode regime
#   (b) include one comm-on control cell to confirm that path still works
#
# Hypotheses each cell probes:
#   A1  baseline             — per-head feed only, no shaping. Floor.
#   A2  aux only             — does the LIAM aux loss alone break OP symmetry?
#   A3  follow only          — does sparse decision-step follow alone do it?
#   A4  match only           — does dense deliberation match alone do it?
#   A5  follow + aux         — follow without the noisy match.
#   A6  replicate-ish escape — m=0.01 f=1.0 a=0.1, now under gaze_mode.
#   A7  stronger match       — same follow+aux, higher match magnitude.
#   A8  weaker follow        — does follow=0.5 still escape with match+aux?
#   A9  stronger follow      — push follow to 2.0.
#   A10 stronger aux         — aux=0.5.
#   B1  comm-on control      — replicate lunar-serenity-1301 to confirm
#                              that the only-known-good config still works.
#
# Shared: per_head=true, gaze_mode=true (gaze=true except B1 which is comm-on),
#         6 seeds, 5M steps. Estimated 2-2.5h/cell.

set -u

NUM_SEEDS=6
TOTAL_STEPS=5e6

run_cell() {
  local gpu="$1"
  local match="$2"
  local follow="$3"
  local aux="$4"
  local label="$5"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} GPU ${gpu} (m=${match} f=${follow} a=${aux} gaze=true)"
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
    algorithm.JA_ATTN_MATCH_COEF=${match} \
    algorithm.JA_ATTN_FOLLOW_COEF=${follow} \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=${aux}
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit ${rc})"
  return ${rc}
}

run_comm_on_control() {
  local gpu="$1"
  local label="$2"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} GPU ${gpu} (COMM ON control, env match=0.05)"
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

# GPU 2: ablations + comm-on control (6 cells)
(
  run_cell 2 0.00 0.0 0.0  "A1_baseline"
  run_cell 2 0.00 0.0 0.1  "A2_aux_only"
  run_cell 2 0.00 1.0 0.0  "A3_follow_only"
  run_cell 2 0.05 0.0 0.0  "A4_match_only"
  run_cell 2 0.00 1.0 0.1  "A5_follow_aux"
  run_comm_on_control 2    "B1_comm_on_control"
) > grid_rich_gpu2.log 2>&1 &
PID_2=$!

# GPU 6: escape replication + magnitude variants (5 cells)
(
  run_cell 6 0.01 1.0 0.1  "A6_escape_replicate"
  run_cell 6 0.05 1.0 0.1  "A7_strong_match"
  run_cell 6 0.05 0.5 0.1  "A8_weak_follow"
  run_cell 6 0.01 2.0 0.1  "A9_strong_follow"
  run_cell 6 0.01 1.0 0.5  "A10_strong_aux"
) > grid_rich_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 2 chain PID: ${PID_2}  (A1..A5 + B1 comm-on control)"
echo "GPU 6 chain PID: ${PID_6}  (A6..A10)"
echo "Logs:"
echo "  tail -f grid_rich_gpu2.log"
echo "  tail -f grid_rich_gpu6.log"
echo "Waiting for both chains to finish..."
wait ${PID_2} ${PID_6}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Grid finished."
