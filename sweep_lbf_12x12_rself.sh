#!/usr/bin/env bash
# 12x12-8food sweep: 3M baseline + best-aux anchor + r_self sweep at aux=0.05.
# 1 seed, 3M steps per cell.
#
# Usage: ./sweep_lbf_12x12_rself.sh [device]   (default device 3)

cd "$(dirname "$0")" || exit 1
device="${1:-3}"

# (aux, r_self) pairs
cells=(
    "0    0      baseline_3M"
    "0.05 0      aux0.05_3M"
    "0.05 0.005  aux0.05_rself0.005"
    "0.05 0.01   aux0.05_rself0.01"
    "0.05 0.02   aux0.05_rself0.02"
)

for cell in "${cells[@]}"; do
    set -- $cell
    aux=$1
    rself=$2
    suffix=$3
    label="lbf12x12-8food_${suffix}"
    echo "================================================================"
    echo "cell: aux=${aux}  r_self=${rself}  (device ${device})  label=${label}"
    echo "================================================================"
    ./run_gpu.sh "${device}" marl.run \
        task=lbf-image-12x12-8food \
        algorithm=ja_ippo_lbf/lbf-image-12x12-8food \
        algorithm.NUM_SEEDS=1 \
        algorithm.TOTAL_TIMESTEPS=3e6 \
        algorithm.JA_AUX_PARTNER_ARGMAX_COEF="${aux}" \
        algorithm.JA_FRUIT_R_SELF_COEF="${rself}" \
        label="${label}"
    echo "cell ${label} finished with exit code $?"
done
