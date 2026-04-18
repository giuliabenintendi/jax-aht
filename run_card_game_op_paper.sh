#!/usr/bin/env bash
# Train the paper-style OP diagnostic on the static card game:
# one focal color pays 0.9, all others pay 1.0.
# Usage: ./run_card_game_op_paper.sh <gpu> [num_seeds] [timesteps]

GPU="${1:?Usage: ./run_card_game_op_paper.sh <gpu> [num_seeds] [timesteps]}"
NUM_SEEDS="${2:-5}"
TOTAL_TIMESTEPS="${3:-5e5}"

mkdir -p logs

nohup ./run_gpu.sh "$GPU" marl.run \
  -cn base_config_ja_ippo \
  task=card-game-op-paper \
  algorithm=ja_ippo/card-game-op-paper \
  algorithm.NUM_SEEDS="$NUM_SEEDS" \
  algorithm.TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
  label=op_paper_focal0_r0.9 \
  > logs/card_game_op_paper.log 2>&1 &

echo "Launched card-game OP paper diagnostic on GPU $GPU"
echo "  num_seeds: $NUM_SEEDS"
echo "  total_timesteps: $TOTAL_TIMESTEPS"
echo "  log: logs/card_game_op_paper.log"
