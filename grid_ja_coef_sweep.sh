#!/usr/bin/env bash
# Coefficient sweep for JA-only without per-step pick-shaping.
# Tests harder/further coefficient combinations than the previous sweep.
# No architectural changes (per-head feed off, embed_dim default).
#
# Common setup (all cells):
#   task=card-game-op-delib-actions (gaze_mode=false, no comm)
#   algorithm.JA_ATTN_SELF_COEF=0       (no own-attention-pick reward)
#   algorithm.JA_GAZE_PICK_COEF=0       (no partner-attention-pick reward)
#   algorithm.JA_PARTNER_FEED_PER_HEAD=false (keep default bandwidth)
#   OP wrappers on (position-shuffle + recolouring)
#
# Cells:
#   GPU 0:
#     A) aux1.0_match0.1            — push aux 2x past previous best
#     C) aux0.5_match0.2_jsd0.2     — heavy card_jsd alongside aux+match
#   GPU 4:
#     B) aux0.5_match0.5            — push match 5x past previous best
#     D) aux2.0_match0.2            — extreme aux only

set -u

NUM_SEEDS=1
TOTAL_STEPS=4e6

run_cell() {
  local gpu="$1"
  local label="$2"
  shift 2
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] starting ${label} on GPU ${gpu}"
  ./run_gpu.sh "${gpu}" marl.run \
    task=card-game-op-delib-actions \
    algorithm=ja_ippo/card-game-op-delib-actions \
    label="${label}" \
    algorithm.NUM_SEEDS=${NUM_SEEDS} \
    algorithm.TOTAL_TIMESTEPS=${TOTAL_STEPS} \
    algorithm.JA_ATTN_SELF_COEF=0.0 \
    algorithm.JA_GAZE_PICK_COEF=0.0 \
    algorithm.JA_PARTNER_FEED_PER_HEAD=false \
    "$@"
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] finished ${label} (exit ${rc})"
  return ${rc}
}

# GPU 0 chain: cell A then cell C
(
  run_cell 0 sweep_aux1.0_match0.1 \
    algorithm.JA_ATTN_MATCH_COEF=0.1 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=1.0 \
    algorithm.JA_CARD_JSD_COEF=0.0

  run_cell 0 sweep_aux0.5_match0.2_jsd0.2 \
    algorithm.JA_ATTN_MATCH_COEF=0.2 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.5 \
    algorithm.JA_CARD_JSD_COEF=0.2
) > sweep_coef_gpu0.log 2>&1 &
GPU0_PID=$!

# GPU 4 chain: cell B then cell D (parallel with GPU 0)
(
  run_cell 4 sweep_aux0.5_match0.5 \
    algorithm.JA_ATTN_MATCH_COEF=0.5 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.5 \
    algorithm.JA_CARD_JSD_COEF=0.0

  run_cell 4 sweep_aux2.0_match0.2 \
    algorithm.JA_ATTN_MATCH_COEF=0.2 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=2.0 \
    algorithm.JA_CARD_JSD_COEF=0.0
) > sweep_coef_gpu4.log 2>&1 &
GPU4_PID=$!

echo "GPU 0 chain PID: ${GPU0_PID}"
echo "GPU 4 chain PID: ${GPU4_PID}"
echo "Cells on GPU 0: sweep_aux1.0_match0.1 then sweep_aux0.5_match0.2_jsd0.2"
echo "Cells on GPU 4: sweep_aux0.5_match0.5 then sweep_aux2.0_match0.2"
echo ""
echo "Tail logs:"
echo "  tail -f sweep_coef_gpu0.log"
echo "  tail -f sweep_coef_gpu4.log"
echo ""
echo "Waiting for both chains to finish..."

wait ${GPU0_PID}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] GPU 0 chain finished."
wait ${GPU4_PID}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] GPU 4 chain finished."
echo "[$(date +%Y-%m-%d_%H:%M:%S)] All cells finished."
