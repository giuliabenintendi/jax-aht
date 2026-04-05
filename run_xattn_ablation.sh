#!/usr/bin/env bash
# Cross-attention ablation on LBF-10food (force_coop=true)
# 3 conditions x 6 seeds each, sequential on one GPU.
#
# Usage: nohup ./run_xattn_ablation.sh 6 > xattn_ablation.log 2>&1 &

set -e
GPU="${1:-6}"
TASK=lbf-image-10food
ALG="ja_ippo/lbf-image-10food"
STEPS=3e6
SEEDS=6
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
  label="xattn_on_b0.001_s${SEEDS}_${DATE}"

# 2) Cross-attention OFF, beta=0.001
echo "[2/3] CROSS_AGENT_ATTN=false, beta=0.001"
./run_gpu.sh "$GPU" marl.run \
  task=$TASK \
  algorithm=$ALG \
  algorithm.TOTAL_TIMESTEPS=$STEPS \
  algorithm.NUM_SEEDS=$SEEDS \
  algorithm.CROSS_AGENT_ATTN=false \
  algorithm.JA_BETA_MAX=0.001 \
  label="xattn_off_b0.001_s${SEEDS}_${DATE}"

# 3) Cross-attention ON, beta=0 (no JSD reward)
echo "[3/3] CROSS_AGENT_ATTN=true, beta=0"
./run_gpu.sh "$GPU" marl.run \
  task=$TASK \
  algorithm=$ALG \
  algorithm.TOTAL_TIMESTEPS=$STEPS \
  algorithm.NUM_SEEDS=$SEEDS \
  algorithm.CROSS_AGENT_ATTN=true \
  algorithm.JA_BETA_MAX=0.0 \
  label="xattn_on_b0.0_s${SEEDS}_${DATE}"

echo "=== All 3 runs complete ==="
echo "Finished: $(date)"
