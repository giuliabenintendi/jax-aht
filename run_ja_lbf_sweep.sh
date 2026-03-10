#!/usr/bin/env bash
# JA-IPPO beta sweep on LBF-image, 1M steps each
DEVICE="${1:-0}"

./run_gpu.sh "$DEVICE" marl.run task=lbf-image algorithm=ja_ippo/lbf-image/default algorithm.TOTAL_TIMESTEPS=1e6 algorithm.JA_BETA=1e-4 label=beta_1e-4 && \
./run_gpu.sh "$DEVICE" marl.run task=lbf-image algorithm=ja_ippo/lbf-image/default algorithm.TOTAL_TIMESTEPS=1e6 algorithm.JA_BETA=1e-3 label=beta_1e-3 && \
./run_gpu.sh "$DEVICE" marl.run task=lbf-image algorithm=ja_ippo/lbf-image/default algorithm.TOTAL_TIMESTEPS=1e6 algorithm.JA_BETA=5e-3 label=beta_5e-3 && \
./run_gpu.sh "$DEVICE" marl.run task=lbf-image algorithm=ja_ippo/lbf-image/default algorithm.TOTAL_TIMESTEPS=1e6 algorithm.JA_BETA=1e-2 label=beta_1e-2 && \
./run_gpu.sh "$DEVICE" marl.run task=lbf-image algorithm=ja_ippo/lbf-image/default algorithm.TOTAL_TIMESTEPS=1e6 algorithm.JA_BETA=5e-2 label=beta_5e-2
