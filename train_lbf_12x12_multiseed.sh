#!/usr/bin/env bash
# 12x12-8food multiseed training for XP comparison.
# One condition at a time.
#
# Usage:
#   ./train_lbf_12x12_multiseed.sh [device] [condition]
#     device    : GPU index (default 3)
#     condition : "baseline" | "aux" | "aux_rself"   (default baseline)
#
# Conditions:
#   baseline           aux=0,    r_self=0,     channel=ON,  NUM_SEEDS=12
#   aux                aux=0.05, r_self=0,     channel=ON,  NUM_SEEDS=12
#   aux_rself          aux=0.05, r_self=0.005, channel=ON,  NUM_SEEDS=12
#   aux_nochannel      aux=0.05, r_self=0,     channel=OFF, NUM_SEEDS=5
#   baseline_nochannel aux=0,    r_self=0,     channel=OFF, NUM_SEEDS=5
#
# Example (parallel on two GPUs):
#   nohup ./train_lbf_12x12_multiseed.sh 3 baseline   > ~/train_gpu3.log 2>&1 &
#   nohup ./train_lbf_12x12_multiseed.sh 4 aux_rself  > ~/train_gpu4.log 2>&1 &

cd "$(dirname "$0")" || exit 1
device="${1:-3}"
condition="${2:-baseline}"

case "${condition}" in
    baseline)
        aux=0
        rself=0
        seeds=12
        label="lbf12x12-8food_baseline_3M_12seed"
        ;;
    aux)
        aux=0.05
        rself=0
        seeds=12
        label="lbf12x12-8food_aux0.05_3M_12seed"
        ;;
    aux_rself)
        aux=0.05
        rself=0.005
        seeds=12
        label="lbf12x12-8food_aux0.05_rself0.005_3M_12seed"
        ;;
    aux_rself_tiny)
        aux=0.05
        rself=0.001
        seeds=12
        label="lbf12x12-8food_aux0.05_rself0.001_3M_12seed"
        ;;
    aux_nochannel)
        aux=0.05
        rself=0
        seeds=5
        channel="False"
        label="lbf12x12-8food_aux0.05_nochannel_3M_5seed"
        ;;
    baseline_nochannel)
        aux=0
        rself=0
        seeds=5
        channel="False"
        label="lbf12x12-8food_baseline_nochannel_3M_5seed"
        ;;
    *)
        echo "unknown condition: ${condition} (expected: baseline | aux | aux_rself | aux_rself_tiny | aux_nochannel | baseline_nochannel)" >&2
        exit 1
        ;;
esac
channel="${channel:-True}"

echo "================================================================"
echo "training: condition=${condition}  aux=${aux} r_self=${rself}  seeds=${seeds}"
echo "device=${device}  label=${label}"
echo "================================================================"

./run_gpu.sh "${device}" marl.run \
    task=lbf-image-12x12-8food \
    algorithm=ja_ippo_lbf/lbf-image-12x12-8food \
    algorithm.NUM_SEEDS="${seeds}" \
    algorithm.TOTAL_TIMESTEPS=3e6 \
    algorithm.JA_AUX_PARTNER_ARGMAX_COEF="${aux}" \
    algorithm.JA_FRUIT_R_SELF_COEF="${rself}" \
    algorithm.JA_FRUIT_PARTNER_FEED="${channel}" \
    label="${label}"
