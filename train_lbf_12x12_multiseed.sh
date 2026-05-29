#!/usr/bin/env bash
# 12x12-8food multiseed training for XP comparison.
# 5 seeds, 3M steps. One condition at a time.
#
# Usage:
#   ./train_lbf_12x12_multiseed.sh [device] [condition]
#     device    : GPU index (default 3)
#     condition : "baseline" or "aux"   (default baseline)
#
# Example (two GPUs in parallel):
#   nohup ./train_lbf_12x12_multiseed.sh 3 baseline > ~/train_gpu3.log 2>&1 &
#   nohup ./train_lbf_12x12_multiseed.sh 4 aux      > ~/train_gpu4.log 2>&1 &

cd "$(dirname "$0")" || exit 1
device="${1:-3}"
condition="${2:-baseline}"

case "${condition}" in
    baseline)
        aux=0
        label="lbf12x12-8food_baseline_3M_5seed"
        ;;
    aux)
        aux=0.05
        label="lbf12x12-8food_aux0.05_3M_5seed"
        ;;
    *)
        echo "unknown condition: ${condition} (expected: baseline or aux)" >&2
        exit 1
        ;;
esac

echo "================================================================"
echo "training: condition=${condition} aux=${aux}  (device ${device})"
echo "label=${label}"
echo "================================================================"

./run_gpu.sh "${device}" marl.run \
    task=lbf-image-12x12-8food \
    algorithm=ja_ippo_lbf/lbf-image-12x12-8food \
    algorithm.NUM_SEEDS=5 \
    algorithm.TOTAL_TIMESTEPS=3e6 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF="${aux}" \
    algorithm.JA_FRUIT_R_SELF_COEF=0.0 \
    label="${label}"
