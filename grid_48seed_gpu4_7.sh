#!/usr/bin/env bash
# Re-run of the 4 GPU-4/7 conditions for the significance paper, 48 seeds.
# (op+comm noshape runs separately on GPU 1 -- NOT part of this script.)
#
#   op + ja           card-game-op-delib-actions  15M  aux 0.25, gaze 0.10, self 0
#   op only           card-game-op                5M   comm off, no JA
#   op + ja shaped    card-game-op-delib-actions  15M  aux 0.25, gaze 0.10, self 0.20
#   op + comm shaped  card-game-op                5M   comm on, match 0.1, follow 0.5
#
#   GPU_A  op+ja        -> op only           ~52h + ~17h
#   GPU_B  op+ja shaped -> op+comm shaped    ~52h + ~17h      (sweep ~69h)
#
# Run from the repo root, inside tmux (stays alive until done):
#   ./grid_48seed_gpu4_7.sh
# Smoke-test the pipeline first (tiny timesteps, minutes):
#   SEEDS=48 STEPS_LONG=1e5 STEPS_SHORT=1e5 ./grid_48seed_gpu4_7.sh
# Override device IDs:
#   GPU_A=0 GPU_B=2 ./grid_48seed_gpu4_7.sh

set -u

GPU_A="${GPU_A:-4}"                          # op+ja -> op only
GPU_B="${GPU_B:-7}"                          # op+ja shaped -> op+comm shaped
SEEDS="${SEEDS:-48}"
STEPS_LONG="${STEPS_LONG:-15e6}"             # op+ja, op+ja shaped
STEPS_SHORT="${STEPS_SHORT:-5e6}"            # op only, op+comm shaped
EVAL_VIDEO_SEEDS="${EVAL_VIDEO_SEEDS:-0}"

[ -f run_gpu.sh ] || { echo "error: run from the repo root (run_gpu.sh not found)"; exit 1; }

ts() { date +%Y-%m-%d_%H:%M:%S; }

echo "[$(ts)] launching 4-run re-sweep (2 GPUs, ${SEEDS} seeds)"
echo "  GPU ${GPU_A}: op+ja -> op only     GPU ${GPU_B}: op+ja shaped -> op+comm shaped"

# --- GPU_A : op + ja  ->  op only -------------------------------------------
(
  echo "[$(ts)] [start]  op+ja  (GPU ${GPU_A})"
  ./run_gpu.sh "${GPU_A}" marl.run \
    task=card-game-op-delib-actions \
    algorithm=ja_ippo/card-game-op-delib-actions \
    label=op_ja_48s \
    algorithm.NUM_SEEDS="${SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS_LONG}" \
    algorithm.JA_ATTN_MATCH_COEF=0.0 \
    algorithm.JA_ATTN_SELF_COEF=0.0 \
    algorithm.JA_GAZE_PICK_COEF=0.10 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.25 \
    algorithm.JA_PARTNER_FEED_PER_HEAD=false \
    algorithm.EVAL_VIDEO_NUM_SEEDS="${EVAL_VIDEO_SEEDS}"
  echo "[$(ts)] [finish] op+ja  (exit $?)"

  echo "[$(ts)] [start]  op only  (GPU ${GPU_A})"
  ./run_gpu.sh "${GPU_A}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    label=op_only_48s \
    algorithm.NUM_SEEDS="${SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS_SHORT}" \
    algorithm.COMMUNICATION=false \
    algorithm.EVAL_VIDEO_NUM_SEEDS="${EVAL_VIDEO_SEEDS}"
  echo "[$(ts)] [finish] op only  (exit $?)"
) > batch48_gpu4.log 2>&1 &
PID_A=$!

# --- GPU_B : op + ja shaped  ->  op + comm shaped ---------------------------
(
  echo "[$(ts)] [start]  op+ja shaped  (GPU ${GPU_B})"
  ./run_gpu.sh "${GPU_B}" marl.run \
    task=card-game-op-delib-actions \
    algorithm=ja_ippo/card-game-op-delib-actions \
    label=op_ja_shaped_48s \
    algorithm.NUM_SEEDS="${SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS_LONG}" \
    algorithm.JA_ATTN_MATCH_COEF=0.0 \
    algorithm.JA_ATTN_SELF_COEF=0.20 \
    algorithm.JA_GAZE_PICK_COEF=0.10 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.25 \
    algorithm.JA_PARTNER_FEED_PER_HEAD=false \
    algorithm.EVAL_VIDEO_NUM_SEEDS="${EVAL_VIDEO_SEEDS}"
  echo "[$(ts)] [finish] op+ja shaped  (exit $?)"

  echo "[$(ts)] [start]  op+comm shaped  (GPU ${GPU_B})"
  ./run_gpu.sh "${GPU_B}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    label=op_comm_shaped_5M_48s \
    algorithm.NUM_SEEDS="${SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS_SHORT}" \
    algorithm.COMMUNICATION=true \
    task.ENV_KWARGS.match_coef=0.1 \
    task.ENV_KWARGS.follow_coef=0.5 \
    task.ENV_KWARGS.stability_coef=0.0 \
    algorithm.EVAL_VIDEO_NUM_SEEDS="${EVAL_VIDEO_SEEDS}"
  echo "[$(ts)] [finish] op+comm shaped  (exit $?)"
) > batch48_gpu7.log 2>&1 &
PID_B=$!

echo "[$(ts)] PIDs:  A=${PID_A} (GPU ${GPU_A})   B=${PID_B} (GPU ${GPU_B})"
echo "[$(ts)] monitor:  tail -f batch48_gpu4.log batch48_gpu7.log"

wait "${PID_A}" "${PID_B}"
echo "[$(ts)] all chains finished."
