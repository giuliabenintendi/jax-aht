#!/usr/bin/env bash
# 5 seeds, 5M, dual critic, no JSD GAE, ent=0.45.
# Coord ring (b0, b0.001) then cramped room (b0, b0.25, b0.5). All sequential.
#
# Usage:
#   ./run_5seed_sweep.sh <gpu>
# Example:
#   ./run_5seed_sweep.sh 2

GPU="${1:?Usage: ./run_5seed_sweep.sh <gpu>}"

TOTAL_TIMESTEPS=5e6
WARMUP_STEPS=3500000
ENT=0.45

mkdir -p logs

nohup bash -c "
echo \"[\$(date +%H:%M)] Starting coord_ring b=0\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/coord_ring \
    algorithm.NUM_SEEDS=5 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS \
    algorithm.JA_WARMUP_ENV_STEPS=$WARMUP_STEPS \
    algorithm.JA_BETA_MAX=0 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=$ENT \
    label=dual_cring_b0_ent045_5s

echo \"[\$(date +%H:%M)] Starting coord_ring b=0.001\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/coord_ring \
    algorithm.NUM_SEEDS=5 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS \
    algorithm.JA_WARMUP_ENV_STEPS=$WARMUP_STEPS \
    algorithm.JA_BETA_MAX=0.001 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=$ENT \
    label=dual_cring_b0001_ent045_5s

echo \"[\$(date +%H:%M)] Starting cramped_room b=0\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.NUM_SEEDS=5 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS \
    algorithm.JA_WARMUP_ENV_STEPS=$WARMUP_STEPS \
    algorithm.JA_BETA_MAX=0 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=$ENT \
    label=dual_cr_b0_ent045_5s

echo \"[\$(date +%H:%M)] Starting cramped_room b=0.25\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.NUM_SEEDS=5 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS \
    algorithm.JA_WARMUP_ENV_STEPS=$WARMUP_STEPS \
    algorithm.JA_BETA_MAX=0.25 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=$ENT \
    label=dual_cr_b025_ent045_5s

echo \"[\$(date +%H:%M)] Starting cramped_room b=0.5\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=overcooked-v1/cramped_room \
    algorithm.NUM_SEEDS=5 \
    algorithm.TRAIN_SEED=42 \
    algorithm.TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS \
    algorithm.JA_WARMUP_ENV_STEPS=$WARMUP_STEPS \
    algorithm.JA_BETA_MAX=0.5 \
    algorithm.USE_DUAL_CRITIC=true \
    algorithm.ENT_COEF=$ENT \
    label=dual_cr_b05_ent045_5s

echo \"[\$(date +%H:%M)] All 5 runs complete\"
" > logs/5seed_sweep.log 2>&1 &

echo "Launched sequential sweep on GPU $GPU (check logs/5seed_sweep.log)"
