#!/usr/bin/env bash
# Quick JA-only sweep to find a config that hits >= 0.30 return without
# per-step pick-shaping (no self, no gaze). 4 cells, 2 per GPU, sequential
# within each GPU, parallel across GPUs. 1 seed, 4M steps each
# (~1.5h per cell, ~3h end-to-end).
#
# Common setup (all cells):
#   task=card-game-op-delib-actions (gaze_mode=false, no comm)
#   algorithm.JA_ATTN_SELF_COEF=0       (no own-attention-pick reward)
#   algorithm.JA_GAZE_PICK_COEF=0       (no partner-attention-pick reward)
#   OP wrappers on (position-shuffle + recolouring)
#
# Cells:
#   GPU 0:
#     1) aux0.5_match0.1            — stronger aux + match, baseline bandwidth
#     2) bandwidth_only             — baseline aux/match, per-head feed + embed=32
#   GPU 4:
#     3) aux0.5_match0.1_jsd0.05    — also turn on the symmetric L2 alignment reward
#     4) max_JA                     — everything: aux=0.5, match=0.1, jsd=0.05,
#                                     per-head, embed=32
#
# Logs:
#   sweep_ja_no_shaping_gpu0.log
#   sweep_ja_no_shaping_gpu4.log

set -u

NUM_SEEDS=1
TOTAL_STEPS=4e6

run_cell() {
  local gpu="$1"
  local label="$2"
  shift 2  # remaining args are config overrides
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] starting ${label} on GPU ${gpu}"
  ./run_gpu.sh "${gpu}" marl.run \
    task=card-game-op-delib-actions \
    algorithm=ja_ippo/card-game-op-delib-actions \
    label="${label}" \
    algorithm.NUM_SEEDS=${NUM_SEEDS} \
    algorithm.TOTAL_TIMESTEPS=${TOTAL_STEPS} \
    algorithm.JA_ATTN_SELF_COEF=0.0 \
    algorithm.JA_GAZE_PICK_COEF=0.0 \
    "$@"
  local rc=$?
  echo "[$(date +%Y-%m-%d_%H:%M:%S)] finished ${label} (exit ${rc})"
  return ${rc}
}

# GPU 0 chain — cells 1 then 2
(
  run_cell 0 sweep_aux0.5_match0.1 \
    algorithm.JA_ATTN_MATCH_COEF=0.1 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.5 \
    algorithm.JA_PARTNER_FEED_PER_HEAD=false

  run_cell 0 sweep_bandwidth_only \
    algorithm.JA_ATTN_MATCH_COEF=0.05 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.25 \
    algorithm.JA_PARTNER_FEED_PER_HEAD=true \
    algorithm.JA_SCALAR_EMBED_DIM=32
) > sweep_ja_no_shaping_gpu0.log 2>&1 &
GPU0_PID=$!

# GPU 4 chain — cells 3 then 4 (parallel with GPU 0)
(
  run_cell 4 sweep_aux0.5_match0.1_jsd0.05 \
    algorithm.JA_ATTN_MATCH_COEF=0.1 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.5 \
    algorithm.JA_CARD_JSD_COEF=0.05 \
    algorithm.JA_PARTNER_FEED_PER_HEAD=false

  run_cell 4 sweep_max_JA \
    algorithm.JA_ATTN_MATCH_COEF=0.1 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.5 \
    algorithm.JA_CARD_JSD_COEF=0.05 \
    algorithm.JA_PARTNER_FEED_PER_HEAD=true \
    algorithm.JA_SCALAR_EMBED_DIM=32
) > sweep_ja_no_shaping_gpu4.log 2>&1 &
GPU4_PID=$!

echo "GPU 0 chain PID: ${GPU0_PID}"
echo "GPU 4 chain PID: ${GPU4_PID}"
echo "Cells on GPU 0: sweep_aux0.5_match0.1 then sweep_bandwidth_only"
echo "Cells on GPU 4: sweep_aux0.5_match0.1_jsd0.05 then sweep_max_JA"
echo ""
echo "Tail logs:"
echo "  tail -f sweep_ja_no_shaping_gpu0.log"
echo "  tail -f sweep_ja_no_shaping_gpu4.log"
echo ""
echo "Waiting for both chains to finish..."

wait ${GPU0_PID}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] GPU 0 chain finished."
wait ${GPU4_PID}
echo "[$(date +%Y-%m-%d_%H:%M:%S)] GPU 4 chain finished."
echo "[$(date +%Y-%m-%d_%H:%M:%S)] All cells finished."
