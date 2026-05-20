#!/usr/bin/env bash
# 48-seed reruns of the 4 card-game conditions for the significance paper.
#
# Conditions kept on their ORIGINAL task configs (decision 2026-05-20):
#   OP only         card-game-op                5M   comm off
#   OP + JA         card-game-op-delib-actions  15M  aux 0.25, gaze 0.10, self 0
#   OP + JA + shap  card-game-op-delib-actions  15M  aux 0.25, gaze 0.10, self 0.20
#   OP + comm       card-game-op                5M   comm on
#
# 3-GPU layout (one chain per GPU, chains run in parallel):
#   GPU_A  OP+JA                ~52h
#   GPU_B  OP+JA+shaping        ~49h
#   GPU_C  OP only -> OP+comm   ~15h + ~15h
# Longest chain ~52h (~2.2 days) -> fits a 3-day weekend.
#
# Seeds run in a sequential loop (ja_ippo.py), constant memory -> NUM_SEEDS=48
# in one job is fine, no OOM, no chunking. All 4 share PRNGKey(42) -> the same
# 48 seeds across conditions (paired).
#
# Run from the repo root, inside tmux/screen (it stays alive until done):
#   ./grid_48seed_significance.sh
# Smoke-test first (quick: all 4 launch at 1 seed / tiny steps):
#   SEEDS=1 STEPS_LONG=1e5 STEPS_SHORT=1e5 ./grid_48seed_significance.sh
# Override device IDs if 1/4/7 are not free:
#   GPU_A=0 GPU_B=2 GPU_C=6 ./grid_48seed_significance.sh

set -u

GPU_A="${GPU_A:-1}"                          # OP+JA
GPU_B="${GPU_B:-4}"                          # OP+JA+shaping
GPU_C="${GPU_C:-7}"                          # OP only, then OP+comm
SEEDS="${SEEDS:-48}"
STEPS_LONG="${STEPS_LONG:-15e6}"             # OP+JA, OP+JA+shaping
STEPS_SHORT="${STEPS_SHORT:-5e6}"            # OP only, OP+comm
EVAL_VIDEO_SEEDS="${EVAL_VIDEO_SEEDS:-0}"    # 0 = no eval videos rendered/saved at all (was 100)

[ -f run_gpu.sh ] || { echo "error: run from the repo root (run_gpu.sh not found)"; exit 1; }

ts() { date +%Y-%m-%d_%H:%M:%S; }

echo "[$(ts)] launching 48-seed significance batch  (SEEDS=${SEEDS})"
echo "  GPU ${GPU_A}: OP+JA    GPU ${GPU_B}: OP+JA+shaping    GPU ${GPU_C}: OP only -> OP+comm"

# --- GPU_A : OP + JA  (self=0) ----------------------------------------------
(
  echo "[$(ts)] [start]  OP+JA  (GPU ${GPU_A})"
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
  echo "[$(ts)] [finish] OP+JA  (exit $?)"
) > batch48_op_ja.log 2>&1 &
PID_A=$!

# --- GPU_B : OP + JA + shaping  (self=0.20) ---------------------------------
(
  echo "[$(ts)] [start]  OP+JA+shaping  (GPU ${GPU_B})"
  ./run_gpu.sh "${GPU_B}" marl.run \
    task=card-game-op-delib-actions \
    algorithm=ja_ippo/card-game-op-delib-actions \
    label=op_ja_shaping_48s \
    algorithm.NUM_SEEDS="${SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS_LONG}" \
    algorithm.JA_ATTN_MATCH_COEF=0.0 \
    algorithm.JA_ATTN_SELF_COEF=0.20 \
    algorithm.JA_GAZE_PICK_COEF=0.10 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.25 \
    algorithm.JA_PARTNER_FEED_PER_HEAD=false \
    algorithm.EVAL_VIDEO_NUM_SEEDS="${EVAL_VIDEO_SEEDS}"
  echo "[$(ts)] [finish] OP+JA+shaping  (exit $?)"
) > batch48_op_ja_shaping.log 2>&1 &
PID_B=$!

# --- GPU_C : OP only  ->  OP + comm -----------------------------------------
(
  echo "[$(ts)] [start]  OP only  (GPU ${GPU_C})"
  ./run_gpu.sh "${GPU_C}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    label=op_only_48s \
    algorithm.NUM_SEEDS="${SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS_SHORT}" \
    algorithm.COMMUNICATION=false \
    algorithm.EVAL_VIDEO_NUM_SEEDS="${EVAL_VIDEO_SEEDS}"
  echo "[$(ts)] [finish] OP only  (exit $?)"

  echo "[$(ts)] [start]  OP+comm  (GPU ${GPU_C})"
  ./run_gpu.sh "${GPU_C}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    label=op_comm_48s \
    algorithm.NUM_SEEDS="${SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS_SHORT}" \
    algorithm.COMMUNICATION=true \
    algorithm.EVAL_VIDEO_NUM_SEEDS="${EVAL_VIDEO_SEEDS}"
  echo "[$(ts)] [finish] OP+comm  (exit $?)"
) > batch48_op_only_comm.log 2>&1 &
PID_C=$!

echo "[$(ts)] PIDs:  A=${PID_A} (OP+JA)   B=${PID_B} (OP+JA+shaping)   C=${PID_C} (OP only->comm)"
echo "[$(ts)] monitor:  tail -f batch48_op_ja.log batch48_op_ja_shaping.log batch48_op_only_comm.log"

wait "${PID_A}" "${PID_B}" "${PID_C}"
echo "[$(ts)] all chains finished."
