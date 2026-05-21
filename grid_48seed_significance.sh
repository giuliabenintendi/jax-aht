#!/usr/bin/env bash
# 5-run card-game sweep for the significance paper.
#
#   op only           card-game-op                5M   48 seeds  comm off, no JA
#   op + ja           card-game-op-delib-actions  15M  48 seeds  aux 0.25, gaze 0.10, self 0
#   op + ja shaped    card-game-op-delib-actions  15M  48 seeds  aux 0.25, gaze 0.10, self 0.20
#   op + comm         card-game-op                15M  12 seeds  comm on, match 0.1, follow 0,   stability 0
#   op + comm shaped  card-game-op                5M   12 seeds  comm on, match 0.1, follow 0.5, stability 0
#
# 2-GPU layout (one chain per GPU, chains run in parallel):
#   GPU_A  op + ja        -> op only                    ~52h + ~15h        = ~67h
#   GPU_B  op + ja shaped -> op+comm -> op+comm shaped   ~49h + ~11h + ~4h  = ~64h
# Longest chain ~67h (~2.8 days) -> fits a 3-day weekend.
#
# Requires the ja_ippo end-of-run OOM fix (commit 0a44124): per-seed train
# outputs are aggregated on the host, so NUM_SEEDS is not GPU-memory-bound.
#
# Run from the repo root, inside tmux/screen (it stays alive until done):
#   ./grid_48seed_significance.sh
# Smoke-test the full pipeline first -- exercises the 48-seed aggregation /
# eval / XP path that OOM'd, with tiny timesteps (minutes, not days):
#   SEEDS=48 STEPS_LONG=1e5 STEPS_SHORT=1e5 ./grid_48seed_significance.sh
# Override device IDs if 4/7 are not free:
#   GPU_A=0 GPU_B=2 ./grid_48seed_significance.sh

set -u

GPU_A="${GPU_A:-4}"                          # op + ja -> op only
GPU_B="${GPU_B:-7}"                          # op + ja shaped -> op+comm -> op+comm shaped
SEEDS="${SEEDS:-48}"                         # op only, op+ja, op+ja shaped
COMM_SEEDS="${COMM_SEEDS:-12}"               # op+comm, op+comm shaped
STEPS_LONG="${STEPS_LONG:-15e6}"             # op+ja, op+ja shaped, op+comm
STEPS_SHORT="${STEPS_SHORT:-5e6}"            # op only, op+comm shaped
EVAL_VIDEO_SEEDS="${EVAL_VIDEO_SEEDS:-0}"    # 0 = no eval videos rendered/saved

[ -f run_gpu.sh ] || { echo "error: run from the repo root (run_gpu.sh not found)"; exit 1; }

ts() { date +%Y-%m-%d_%H:%M:%S; }

echo "[$(ts)] launching 5-run card-game sweep (2 GPUs)"
echo "  GPU ${GPU_A}: op+ja -> op only    GPU ${GPU_B}: op+ja shaped -> op+comm -> op+comm shaped"

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
) > batch48_ja_oponly.log 2>&1 &
PID_A=$!

# --- GPU_B : op + ja shaped  ->  op + comm  ->  op + comm shaped -------------
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

  echo "[$(ts)] [start]  op+comm  (GPU ${GPU_B})"
  ./run_gpu.sh "${GPU_B}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    label=op_comm_noshape_15M_12s \
    algorithm.NUM_SEEDS="${COMM_SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS_LONG}" \
    algorithm.COMMUNICATION=true \
    task.ENV_KWARGS.match_coef=0.1 \
    task.ENV_KWARGS.follow_coef=0.0 \
    task.ENV_KWARGS.stability_coef=0.0 \
    algorithm.EVAL_VIDEO_NUM_SEEDS="${EVAL_VIDEO_SEEDS}"
  echo "[$(ts)] [finish] op+comm  (exit $?)"

  echo "[$(ts)] [start]  op+comm shaped  (GPU ${GPU_B})"
  ./run_gpu.sh "${GPU_B}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    label=op_comm_shaped_5M_12s \
    algorithm.NUM_SEEDS="${COMM_SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS_SHORT}" \
    algorithm.COMMUNICATION=true \
    task.ENV_KWARGS.match_coef=0.1 \
    task.ENV_KWARGS.follow_coef=0.5 \
    task.ENV_KWARGS.stability_coef=0.0 \
    algorithm.EVAL_VIDEO_NUM_SEEDS="${EVAL_VIDEO_SEEDS}"
  echo "[$(ts)] [finish] op+comm shaped  (exit $?)"
) > batch48_jashaped_comm.log 2>&1 &
PID_B=$!

echo "[$(ts)] PIDs:  A=${PID_A} (GPU ${GPU_A})   B=${PID_B} (GPU ${GPU_B})"
echo "[$(ts)] monitor:  tail -f batch48_ja_oponly.log batch48_jashaped_comm.log"

wait "${PID_A}" "${PID_B}"
echo "[$(ts)] all chains finished."
