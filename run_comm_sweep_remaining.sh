#!/usr/bin/env bash
# Communication sweep — remaining 23 runs.
# Usage: ./run_comm_sweep_remaining.sh <gpu> <group>
#   group 1: ent 0.3 (5 remaining) + ent 0.5 (6) = 11 runs
#   group 2: ent 1.0 (6) + ent 2.0 (6) = 12 runs

GPU="${1:?Usage: ./run_comm_sweep_remaining.sh <gpu> <group>}"
GROUP="${2:?Usage: ./run_comm_sweep_remaining.sh <gpu> <group>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game algorithm.NUM_SEEDS=5 algorithm.TOTAL_TIMESTEPS=2.5e5 algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=125000 algorithm.COMMUNICATION=true algorithm.FEED_OTHER_ATTN=true"

if [ "$GROUP" = "1" ]; then
nohup bash -c "
# ent=0.3 remaining (b0.0 already running, skip it)
for BETA in 0.5 1.0 5.0 10.0; do
  echo \"[\$(date +%H:%M)] ent=0.3 beta=\$BETA\"
  $CMD algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=\$BETA label=comm_sweep_ent0.3_b\${BETA}_5s
done
# ent=0.5 all 6
for BETA in 0.0 0.25 0.5 1.0 5.0 10.0; do
  echo \"[\$(date +%H:%M)] ent=0.5 beta=\$BETA\"
  $CMD algorithm.ENT_COEF=0.5 algorithm.JA_BETA_MAX=\$BETA label=comm_sweep_ent0.5_b\${BETA}_5s
done
echo \"[\$(date +%H:%M)] Group 1 Done\"
" > logs/comm_remaining_g1.log 2>&1 &
echo "Launched group 1 (ent 0.3+0.5, 10 runs) on GPU $GPU"

elif [ "$GROUP" = "2" ]; then
nohup bash -c "
for ENT in 1.0 2.0; do
for BETA in 0.0 0.25 0.5 1.0 5.0 10.0; do
  echo \"[\$(date +%H:%M)] ent=\$ENT beta=\$BETA\"
  $CMD algorithm.ENT_COEF=\$ENT algorithm.JA_BETA_MAX=\$BETA label=comm_sweep_ent\${ENT}_b\${BETA}_5s
done; done
echo \"[\$(date +%H:%M)] Group 2 Done\"
" > logs/comm_remaining_g2.log 2>&1 &
echo "Launched group 2 (ent 1.0+2.0, 12 runs) on GPU $GPU"

else
echo "Unknown group: $GROUP (use 1-2)"
exit 1
fi
