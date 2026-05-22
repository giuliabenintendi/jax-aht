#!/usr/bin/env bash
# Sweep JA_AUX_PARTNER_ARGMAX_COEF on the OP+JA+shaping recipe (run 0x6s07vx).
# Holds the shaping recipe fixed (self=0.20, gaze=0.10, match=0); varies only aux.
# 1 seed, 15M steps per value. XP eval is skipped automatically (num_seeds == 1).
#
# Usage: ./sweep_aux.sh [device]   (default device 0)

cd "$(dirname "$0")" || exit 1
device="${1:-0}"
aux_values=(0 0.001 0.003 0.01 0.03 0.1)

for aux in "${aux_values[@]}"; do
    echo "================================================================"
    echo "aux sweep: JA_AUX_PARTNER_ARGMAX_COEF=${aux}  (device ${device})"
    echo "================================================================"
    ./run_gpu.sh "${device}" marl.run \
        task=card-game-op-delib-actions \
        algorithm=ja_ippo/card-game-op-delib-actions \
        label="aux_sweep/aux${aux}_1s" \
        algorithm.NUM_SEEDS=1 \
        algorithm.TOTAL_TIMESTEPS=15e6 \
        algorithm.JA_ATTN_MATCH_COEF=0.0 \
        algorithm.JA_ATTN_SELF_COEF=0.20 \
        algorithm.JA_GAZE_PICK_COEF=0.10 \
        algorithm.JA_AUX_PARTNER_ARGMAX_COEF="${aux}" \
        algorithm.JA_PARTNER_FEED_PER_HEAD=false
    echo "aux=${aux} finished with exit code $?"
done
