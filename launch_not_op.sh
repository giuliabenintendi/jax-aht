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
SEEDS=${2:-12}
STEPS=${3:-5e6}

# Default 12 seeds: NOT-OP is a near-deterministic baseline (each seed locks
# to a private color convention; SP=1.0, XP=0.2 by chance match), so more
# seeds wouldn't tighten the curves materially. 12 gives a clean Hu-style
# learning curve in ~1.5-2.5h.

./run_gpu.sh "$GPU" marl.run \
    task=card-game \
    algorithm=ja_ippo/card-game-op \
    label=not_op_${SEEDS}s_5M \
    algorithm.NUM_SEEDS="$SEEDS" \
    algorithm.TOTAL_TIMESTEPS="$STEPS" \
    algorithm.COMMUNICATION=false \
    algorithm.EVAL_VIDEO_NUM_SEEDS=0 \
    algorithm.JA_CARD_METRIC=false \
    algorithm.JA_CARD_ATTN=false \
    algorithm.JA_CARD_PARTNER_FEED=false
