#!/usr/bin/env bash
# OP+comm card-game conditions, chained to start after the OP-only job finishes.
#
#   comm WITHOUT shaping   match 0.1, follow 0,   stability 0    15M
#   comm WITH shaping      match 0.1, follow 0.5, stability 0     5M
#
# Both: COMMUNICATION=true, task card-game-op, 12 seeds. They differ only in
# the follow term and the step budget. comm-without-shaping gets 15M to match
# the OP+JA base budget (it is the harder condition; the match-only run
# 561ukjtx left 2/3 seeds at chance at 10M). comm-with-shaping stays at 5M:
# it is already saturated there (c77tmvf3 reached 0.91 at 5M).
#
# Run from the repo root inside tmux/screen.
#   ./run_comm_conditions.sh
# Chain after a running OP-only job by passing its PID (same machine):
#   WAIT_PID=<pid> ./run_comm_conditions.sh
# Override GPU / seeds:
#   GPU=2 SEEDS=12 ./run_comm_conditions.sh

set -u

GPU="${GPU:-0}"
SEEDS="${SEEDS:-12}"
WAIT_PID="${WAIT_PID:-}"

[ -f run_gpu.sh ] || { echo "error: run from the repo root (run_gpu.sh not found)"; exit 1; }

ts() { date +%Y-%m-%d_%H:%M:%S; }

if [ -n "${WAIT_PID}" ]; then
  echo "[$(ts)] waiting for OP-only (PID ${WAIT_PID}) to finish..."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
  echo "[$(ts)] OP-only finished."
fi

echo "[$(ts)] [start]  comm WITHOUT shaping  (15M, GPU ${GPU})"
./run_gpu.sh "${GPU}" marl.run \
  task=card-game-op \
  algorithm=ja_ippo/card-game-op \
  label=op_comm_noshape_15M_${SEEDS}s \
  algorithm.NUM_SEEDS="${SEEDS}" \
  algorithm.TOTAL_TIMESTEPS=15e6 \
  algorithm.COMMUNICATION=true \
  algorithm.EVAL_VIDEO_NUM_SEEDS=0 \
  task.ENV_KWARGS.match_coef=0.1 \
  task.ENV_KWARGS.follow_coef=0.0 \
  task.ENV_KWARGS.stability_coef=0.0
echo "[$(ts)] [finish] comm WITHOUT shaping  (exit $?)"

echo "[$(ts)] [start]  comm WITH shaping  (5M, GPU ${GPU})"
./run_gpu.sh "${GPU}" marl.run \
  task=card-game-op \
  algorithm=ja_ippo/card-game-op \
  label=op_comm_shape_5M_${SEEDS}s \
  algorithm.NUM_SEEDS="${SEEDS}" \
  algorithm.TOTAL_TIMESTEPS=5e6 \
  algorithm.COMMUNICATION=true \
  algorithm.EVAL_VIDEO_NUM_SEEDS=0 \
  task.ENV_KWARGS.match_coef=0.1 \
  task.ENV_KWARGS.follow_coef=0.5 \
  task.ENV_KWARGS.stability_coef=0.0
echo "[$(ts)] [finish] comm WITH shaping  (exit $?)"

echo "[$(ts)] all comm conditions finished."
