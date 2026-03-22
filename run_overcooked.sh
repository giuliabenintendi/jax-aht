#!/usr/bin/env bash
# Overcooked experiments. Edit the runs below as needed.
# Usage: ./run_overcooked.sh <gpu>
# Example: ./run_overcooked.sh 2

GPU="${1:?Usage: ./run_overcooked.sh <gpu>}"

mkdir -p logs

nohup bash -c "

echo \"[\$(date +%H:%M)] forced_coord 8M b0.5 dual ent=0.4 5s\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/forced_coord \
    algorithm=ja_ippo/overcooked-v1/forced_coord \
    algorithm.NUM_SEEDS=5 \
    algorithm.TOTAL_TIMESTEPS=8e6 \
    algorithm.JA_WARMUP_ENV_STEPS=5600000 \
    algorithm.JA_BETA_MAX=0.5 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=0.4 \
    label=dual_fc_b05_ent040_8M_5s

echo \"[\$(date +%H:%M)] All overcooked runs complete\"
" > logs/overcooked.log 2>&1 &

echo "Launched overcooked experiments on GPU $GPU (check logs/overcooked.log)"
