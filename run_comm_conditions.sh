#!/usr/bin/env bash
# OP / comm card-game baseline conditions at 48 seeds.
#
#   op     OP only             COMMUNICATION=false
#   full   comm fully shaped   match 0.1, follow 0.5, stability 0
#   match  comm match-only     match 0.1, follow 0,   stability 0
#
# All three: task card-game-op, 48 seeds, 5M steps.
#
# NOTE: match-only is the hard condition -- run 561ukjtx left 2/3 seeds at
# chance at 10M. The 5M budget here is a deliberate, confirmed choice, not an
# oversight; expect some seeds to finish at chance.
#
# Run from the repo root inside tmux/screen:
#   ./run_comm_conditions.sh
# Run one condition only (to parallelize across GPUs):
#   COND=op ./run_comm_conditions.sh
# Override GPU / seeds / steps:
#   GPU=5 SEEDS=48 STEPS=5e6 ./run_comm_conditions.sh
# Chain after a running job on the same machine:
#   WAIT_PID=<pid> ./run_comm_conditions.sh

set -u

GPU="${GPU:-5}"
SEEDS="${SEEDS:-48}"
STEPS="${STEPS:-5e6}"
COND="${COND:-all}"
WAIT_PID="${WAIT_PID:-}"

[ -f run_gpu.sh ] || { echo "error: run from the repo root (run_gpu.sh not found)"; exit 1; }

ts() { date +%Y-%m-%d_%H:%M:%S; }

if [ -n "${WAIT_PID}" ]; then
  echo "[$(ts)] waiting for PID ${WAIT_PID} to finish..."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
  echo "[$(ts)] PID ${WAIT_PID} finished."
fi

run_op() {
  echo "[$(ts)] [start]  OP only  (${STEPS}, GPU ${GPU})"
  ./run_gpu.sh "${GPU}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    label=comm_rerun/op_only_${SEEDS}s \
    algorithm.NUM_SEEDS="${SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS}" \
    algorithm.COMMUNICATION=false \
    algorithm.EVAL_VIDEO_NUM_SEEDS=0
  echo "[$(ts)] [finish] OP only  (exit $?)"
}

run_full() {
  echo "[$(ts)] [start]  comm fully shaped  (${STEPS}, GPU ${GPU})"
  ./run_gpu.sh "${GPU}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    label=comm_rerun/comm_full_${SEEDS}s \
    algorithm.NUM_SEEDS="${SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS}" \
    algorithm.COMMUNICATION=true \
    algorithm.EVAL_VIDEO_NUM_SEEDS=0 \
    task.ENV_KWARGS.match_coef=0.1 \
    task.ENV_KWARGS.follow_coef=0.5 \
    task.ENV_KWARGS.stability_coef=0.0
  echo "[$(ts)] [finish] comm fully shaped  (exit $?)"
}

run_match() {
  echo "[$(ts)] [start]  comm match-only  (${STEPS}, GPU ${GPU})"
  ./run_gpu.sh "${GPU}" marl.run \
    task=card-game-op \
    algorithm=ja_ippo/card-game-op \
    label=comm_rerun/comm_matchonly_${SEEDS}s \
    algorithm.NUM_SEEDS="${SEEDS}" \
    algorithm.TOTAL_TIMESTEPS="${STEPS}" \
    algorithm.COMMUNICATION=true \
    algorithm.EVAL_VIDEO_NUM_SEEDS=0 \
    task.ENV_KWARGS.match_coef=0.1 \
    task.ENV_KWARGS.follow_coef=0.0 \
    task.ENV_KWARGS.stability_coef=0.0
  echo "[$(ts)] [finish] comm match-only  (exit $?)"
}

echo "[$(ts)] GPU=${GPU} SEEDS=${SEEDS} STEPS=${STEPS} COND=${COND}"
case "${COND}" in
  op)    run_op ;;
  full)  run_full ;;
  match) run_match ;;
  all)   run_op; run_full; run_match ;;
  *)     echo "unknown COND: ${COND} (use op|full|match|all)"; exit 1 ;;
esac
echo "[$(ts)] done."
