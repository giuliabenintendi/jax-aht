#!/usr/bin/env bash
# Run feed_other_attn experiments on cramped_room.
# Usage: ./run_feed_attn.sh <gpu_id>

GPU="${1:-0}"
TIMESTEPS=5e6

echo "=== Feed attn, beta=0: cramped_room ==="
./run_gpu.sh "$GPU" marl.run \
    algorithm=ja_ippo/overcooked-v1/cramped_room \
    task=overcooked-v1/cramped_room \
    algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
    algorithm.FEED_OTHER_ATTN=true \
    algorithm.JA_BETA_MAX=0.0 \
    label=feed_attn_beta0

echo "=== Feed attn, beta=0.25: cramped_room ==="
./run_gpu.sh "$GPU" marl.run \
    algorithm=ja_ippo/overcooked-v1/cramped_room \
    task=overcooked-v1/cramped_room \
    algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
    algorithm.FEED_OTHER_ATTN=true \
    algorithm.JA_BETA_MAX=0.25 \
    label=feed_attn_beta025

echo "All runs complete."
