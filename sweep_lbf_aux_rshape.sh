#!/usr/bin/env bash
# 2x2 sweep over JA_AUX_PARTNER_ARGMAX_COEF x JA_FRUIT_R_SHAPE_COEF on
# lbf-image-12x12-8food. 1 seed, 1M steps per cell.
#
# Usage: ./sweep_lbf_aux_rshape.sh [device]   (default device 3)

cd "$(dirname "$0")" || exit 1
device="${1:-3}"

# (aux, r_shape) pairs
cells=(
    "0.05 0.02"
    "0.05 0.1"
    "0.2  0.02"
    "0.2  0.1"
)

for cell in "${cells[@]}"; do
    set -- $cell
    aux=$1
    r=$2
    label="lbf12x12-8food_aux${aux}_r${r}"
    echo "================================================================"
    echo "cell: aux=${aux} r=${r}  (device ${device})  label=${label}"
    echo "================================================================"
    ./run_gpu.sh "${device}" marl.run \
        task=lbf-image-12x12-8food \
        algorithm=ja_ippo_lbf/lbf-image-12x12-8food \
        algorithm.NUM_SEEDS=1 \
        algorithm.TOTAL_TIMESTEPS=1e6 \
        algorithm.JA_AUX_PARTNER_ARGMAX_COEF="${aux}" \
        algorithm.JA_FRUIT_R_SHAPE_COEF="${r}" \
        label="${label}"
    echo "cell aux=${aux} r=${r} finished with exit code $?"
done
