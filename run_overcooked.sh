#!/usr/bin/env bash
# Overcooked experiments. Edit the runs below as needed.
# Usage: ./run_overcooked.sh <gpu>

GPU="${1:?Usage: ./run_overcooked.sh <gpu>}"

mkdir -p logs

nohup bash -c "

echo \"[\$(date +%H:%M)] forced_coord dual b=0 ent=0.4 8M 5s\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/forced_coord \
    algorithm=ja_ippo/overcooked-v1/forced_coord \
    algorithm.NUM_SEEDS=5 \
    algorithm.TOTAL_TIMESTEPS=8e6 \
    algorithm.JA_WARMUP_ENV_STEPS=5600000 \
    algorithm.JA_BETA_MAX=0.0 \
    algorithm.ENT_COEF=0.4 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.DUAL_CRITIC_ACTOR_JA=false

echo \"[\$(date +%H:%M)] Done\"
" > logs/overcooked.log 2>&1 &

echo "Launched overcooked experiments on GPU $GPU (check logs/overcooked.log)"
