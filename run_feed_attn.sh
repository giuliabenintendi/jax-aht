#!/usr/bin/env bash
# Run feed_other_attn experiments on cramped_room.
# Usage: ./run_feed_attn.sh <gpu>
# Example: ./run_feed_attn.sh 6

GPU="${1:?Usage: ./run_feed_attn.sh <gpu>}"
TIMESTEPS=5e6

mkdir -p logs

nohup bash -c "
echo \"[\$(date +%H:%M)] Starting feed_attn beta=0, cramped_room\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
    algorithm.FEED_OTHER_ATTN=true \
    algorithm.JA_BETA_MAX=0.0 \
    label=feed_attn_beta0

echo \"[\$(date +%H:%M)] Starting feed_attn beta=0.25, cramped_room\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
    algorithm.FEED_OTHER_ATTN=true \
    algorithm.JA_BETA_MAX=0.25 \
    label=feed_attn_beta025

echo \"[\$(date +%H:%M)] All runs complete\"
" > logs/feed_attn.log 2>&1 &

echo "Launched feed_attn sweep on GPU $GPU (check logs/feed_attn.log)"
