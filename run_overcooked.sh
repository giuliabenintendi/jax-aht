#!/usr/bin/env bash
# Overcooked experiments. Edit the runs below as needed.
# Usage: ./run_overcooked.sh <gpu>

GPU="${1:?Usage: ./run_overcooked.sh <gpu>}"

mkdir -p logs

nohup bash -c "

echo \"[\$(date +%H:%M)] coord_ring single b=0.001 ent=0.01 5s\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/coord_ring \
    algorithm=ja_ippo/overcooked-v1/coord_ring \
    algorithm.NUM_SEEDS=5 \
    algorithm.TOTAL_TIMESTEPS=5e6 \
    algorithm.JA_BETA_MAX=0.001

echo \"[\$(date +%H:%M)] coord_ring single b=0.001 ent=0.45 5s\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/coord_ring \
    algorithm=ja_ippo/overcooked-v1/coord_ring \
    algorithm.NUM_SEEDS=5 \
    algorithm.TOTAL_TIMESTEPS=5e6 \
    algorithm.JA_BETA_MAX=0.001 \
    algorithm.ENT_COEF=0.45

echo \"[\$(date +%H:%M)] Done\"
" > logs/overcooked.log 2>&1 &

echo "Launched overcooked experiments on GPU $GPU (check logs/overcooked.log)"
