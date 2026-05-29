#!/usr/bin/env bash
# Single-cell run on lbf-image-12x12-8food:
#   aux=0.05, r_shape=0.01, 5 seeds, 1M steps each.
#
# Usage: ./run_lbf_aux0.05_r0.01.sh [device]   (default device 4)

cd "$(dirname "$0")" || exit 1
device="${1:-4}"

./run_gpu.sh "${device}" marl.run \
    task=lbf-image-12x12-8food \
    algorithm=ja_ippo_lbf/lbf-image-12x12-8food \
    algorithm.NUM_SEEDS=5 \
    algorithm.TOTAL_TIMESTEPS=1e6 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF=0.05 \
    algorithm.JA_FRUIT_R_SHAPE_COEF=0.01 \
    label="lbf12x12-8food_aux0.05_r0.01_5seed"
