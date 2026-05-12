#!/usr/bin/env bash
# Rich grid sweep across (match, follow, aux) — hunt for escape seeds.
#
# Hypotheses each cell probes:
#   A1  baseline           — per-head feed only, no shaping. Floor reference.
#   A2  aux only           — does the LIAM aux loss alone break OP symmetry?
#   A3  follow only        — does sparse decision-step follow alone do it?
#   A4  match only         — does dense deliberation-step match alone do it?
#   A5  follow + aux       — follow without the noisy match coefficient.
#   A6  REPLICATE escape   — yesterday's m=0.01 f=1.0 a=0.1 seed-3 0.36 config.
#   A7  stronger match     — same follow+aux, higher match magnitude.
#   A8  weaker follow      — does follow=0.5 still escape with match+aux?
#   A9  stronger follow    — push follow to 2.0 — does it overpower env reward?
#   A10 stronger aux       — aux=0.5 — does aux loss matter more when amplified?
#
# All cells: per_head=true, gaze=false, COMMUNICATION=false, 6 seeds, 5M steps.
# Estimated wall time: ~2-2.5h/cell  ->  5 cells x 2 GPUs ~= 10-12h overnight.

set -u

NUM_SEEDS=6
TOTAL_STEPS=5e6

run_cell() {
  local gpu="$1"
  local match="$2"
  local follow="$3"
  local aux="$4"
  local label="$5"
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [start] ${label} GPU ${gpu} (m=${match} f=${follow} a=${aux})"
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
    algorithm.JA_ATTN_FOLLOW_COEF=${follow} \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=${aux}
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] [finish] ${label} (exit ${rc})"
  return ${rc}
}

# GPU 2: ablations + escape replication (5 cells)
(
  run_cell 2 0.00 0.0 0.0  "A1_baseline"
  run_cell 2 0.00 0.0 0.1  "A2_aux_only"
  run_cell 2 0.00 1.0 0.0  "A3_follow_only"
  run_cell 2 0.05 0.0 0.0  "A4_match_only"
  run_cell 2 0.00 1.0 0.1  "A5_follow_aux"
) > grid_rich_gpu2.log 2>&1 &
PID_2=$!

# GPU 6: replicate escape + magnitude variants (5 cells)
(
  run_cell 6 0.01 1.0 0.1  "A6_escape_replicate"
  run_cell 6 0.05 1.0 0.1  "A7_strong_match"
  run_cell 6 0.05 0.5 0.1  "A8_weak_follow"
  run_cell 6 0.01 2.0 0.1  "A9_strong_follow"
  run_cell 6 0.01 1.0 0.5  "A10_strong_aux"
) > grid_rich_gpu6.log 2>&1 &
PID_6=$!

echo "GPU 2 chain PID: ${PID_2}  (A1..A5)"
echo "GPU 6 chain PID: ${PID_6}  (A6..A10)"
echo "Logs:"
echo "  tail -f grid_rich_gpu2.log"
echo "  tail -f grid_rich_gpu6.log"
echo "Waiting for both chains to finish..."
wait ${PID_2} ${PID_6}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] Grid finished."
