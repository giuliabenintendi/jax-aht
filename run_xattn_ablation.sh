#!/usr/bin/env bash
# Cross-attention (gated fusion) ablation on LBF-10food (force_coop=true)
# Sequential on one GPU.
#
# Already have:
#   063tp01k: xattn OFF, beta=0.001 (baseline)
#   euhag8e6: xattn ON,  beta=0.0
#
# This script runs:
#   1) xattn ON,  beta=0.001
#   2) xattn ON,  beta=0.002
#   3) xattn ON,  beta=0.005
#
# Warmup is set to 2M (< 3M total) so beta actually reaches its max.
#
# Usage: nohup ./run_xattn_ablation.sh 6 > xattn_ablation.log 2>&1 &

set -e
GPU="${1:-6}"
TASK=lbf-image-10food
ALG="ja_ippo/lbf-image-10food"
STEPS=3e6
SEEDS=6
WARMUP=2e6
DATE=$(date +%d%m%Y)

echo "=== Cross-attention ablation on GPU ${GPU} ==="
echo "Started: $(date)"

# 1) Cross-attention ON, beta=0.001
echo "[1/3] CROSS_AGENT_ATTN=true, beta=0.001"
./run_gpu.sh "$GPU" marl.run \
  task=$TASK \
  algorithm=$ALG \
  algorithm.TOTAL_TIMESTEPS=$STEPS \
  algorithm.NUM_SEEDS=$SEEDS \
  algorithm.CROSS_AGENT_ATTN=true \
  algorithm.JA_BETA_MAX=0.001 \
  algorithm.JA_WARMUP_ENV_STEPS=$WARMUP \
  label="xattn_on_b0.001_s${SEEDS}_${DATE}"

# 2) Cross-attention ON, beta=0.002
echo "[2/3] CROSS_AGENT_ATTN=true, beta=0.002"
./run_gpu.sh "$GPU" marl.run \
  task=$TASK \
  algorithm=$ALG \
  algorithm.TOTAL_TIMESTEPS=$STEPS \
  algorithm.NUM_SEEDS=$SEEDS \
  algorithm.CROSS_AGENT_ATTN=true \
  algorithm.JA_BETA_MAX=0.002 \
  algorithm.JA_WARMUP_ENV_STEPS=$WARMUP \
  label="xattn_on_b0.002_s${SEEDS}_${DATE}"

# 3) Cross-attention ON, beta=0.005
echo "[3/3] CROSS_AGENT_ATTN=true, beta=0.005"
./run_gpu.sh "$GPU" marl.run \
  task=$TASK \
  algorithm=$ALG \
  algorithm.TOTAL_TIMESTEPS=$STEPS \
  algorithm.NUM_SEEDS=$SEEDS \
  algorithm.CROSS_AGENT_ATTN=true \
  algorithm.JA_BETA_MAX=0.005 \
  algorithm.JA_WARMUP_ENV_STEPS=$WARMUP \
  label="xattn_on_b0.005_s${SEEDS}_${DATE}"

echo "=== All 3 runs complete ==="
echo "Finished: $(date)"
