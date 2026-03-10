#!/usr/bin/env bash
# Beta sweep for JA-IPPO on LBF-image with 10 food.
# 4 beta values on GPU 6.
#
# Usage: ./run_lbf_beta_sweep.sh [gpu_device]
#   Default GPU: 6

set -e

GPU="${1:-6}"
TIMESTEPS=5e6
WARMUP=3500000
BETAS=(1e-3 5e-3 8e-3 0.01)

for BETA in "${BETAS[@]}"; do
    echo "=== LBF-10food BETA=${BETA} ==="
    ./run_gpu.sh "$GPU" marl.run \
        -cn base_config_ja_ippo \
        algorithm=ja_ippo/lbf-image-10food \
        task=lbf-image-10food \
        algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
        algorithm.JA_WARMUP_ENV_STEPS=$WARMUP \
        algorithm.JA_BETA_MAX=$BETA \
        label=beta_sweep
done

echo "All LBF sweep runs complete."
