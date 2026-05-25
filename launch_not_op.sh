#!/usr/bin/env bash
# Train the NOT-OP (plain self-play, no OP wrappers) baseline for the
# Hu et al. style over-training figure.
#
# Each seed converges to a private color/position convention; SP perfect, XP at
# chance (1/5 = 0.2) because two different seeds pick different conventions.
#
# Usage:
#   bash launch_not_op.sh [GPU=1] [SEEDS=48] [STEPS=5e6]
set -euo pipefail

GPU=${1:-1}
SEEDS=${2:-48}
STEPS=${3:-5e6}

./run_gpu.sh "$GPU" marl.run \
    task=card-game \
    algorithm=ja_ippo/card-game-op \
    label=not_op_48s_5M \
    algorithm.NUM_SEEDS="$SEEDS" \
    algorithm.TOTAL_TIMESTEPS="$STEPS" \
    algorithm.COMMUNICATION=false \
    algorithm.EVAL_VIDEO_NUM_SEEDS=0
