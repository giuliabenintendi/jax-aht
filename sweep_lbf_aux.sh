#!/usr/bin/env bash
# Aux-only sweep on lbf-image-12x12-8food. r_self held at 0.
# 1 seed, 1M steps per cell. aux in {0, 0.01, 0.05, 0.1}.
#
# Usage: ./sweep_lbf_aux.sh [device]   (default device 3)

cd "$(dirname "$0")" || exit 1
device="${1:-3}"

aux_values=(0 0.01 0.05 0.1)

for aux in "${aux_values[@]}"; do
    label="lbf12x12-8food_aux${aux}_rself0"
    echo "================================================================"
    echo "aux=${aux}  r_self=0  (device ${device})  label=${label}"
    echo "================================================================"
    ./run_gpu.sh "${device}" marl.run \
        task=lbf-image-12x12-8food \
        algorithm=ja_ippo_lbf/lbf-image-12x12-8food \
        algorithm.NUM_SEEDS=1 \
        algorithm.TOTAL_TIMESTEPS=1e6 \
        algorithm.JA_AUX_PARTNER_ARGMAX_COEF="${aux}" \
        algorithm.JA_FRUIT_R_SELF_COEF=0.0 \
        label="${label}"
    echo "aux=${aux} finished with exit code $?"
done
