#!/usr/bin/env bash
# Communication sweep. 250k steps, 5 seeds, comm=on, feed=on.
# 42 runs: 7 entropy x 6 beta.
# Usage: ./run_comm_sweep.sh <gpu> <group>
#   group 1 (GPU 2): ent 0.1, 0.3, 0.5 x 6 betas = 18 runs ~7.5h
#   group 2 (GPU 3): ent 1.0, 2.0 x 6 betas = 12 runs ~5h
#   group 3 (GPU 5): ent 5.0, 10.0 x 6 betas = 12 runs ~5h

GPU="${1:?Usage: ./run_comm_sweep.sh <gpu> <group>}"
GROUP="${2:?Usage: ./run_comm_sweep.sh <gpu> <group>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game algorithm.NUM_SEEDS=5 algorithm.TOTAL_TIMESTEPS=2.5e5 algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=125000 algorithm.COMMUNICATION=true algorithm.FEED_OTHER_ATTN=true"

if [ "$GROUP" = "1" ]; then
nohup bash -c "
for ENT in 0.1 0.3 0.5; do
for BETA in 0.0 0.25 0.5 1.0 5.0 10.0; do
  echo \"[\$(date +%H:%M)] ent=\$ENT beta=\$BETA\"
  $CMD algorithm.ENT_COEF=\$ENT algorithm.JA_BETA_MAX=\$BETA label=comm_sweep_ent\${ENT}_b\${BETA}_5s
done; done
echo \"[\$(date +%H:%M)] Group 1 Done\"
" > logs/comm_sweep_g1.log 2>&1 &
echo "Launched group 1 (ent 0.1/0.3/0.5, 18 runs) on GPU $GPU"

elif [ "$GROUP" = "2" ]; then
nohup bash -c "
for ENT in 1.0 2.0; do
for BETA in 0.0 0.25 0.5 1.0 5.0 10.0; do
  echo \"[\$(date +%H:%M)] ent=\$ENT beta=\$BETA\"
  $CMD algorithm.ENT_COEF=\$ENT algorithm.JA_BETA_MAX=\$BETA label=comm_sweep_ent\${ENT}_b\${BETA}_5s
done; done
echo \"[\$(date +%H:%M)] Group 2 Done\"
" > logs/comm_sweep_g2.log 2>&1 &
echo "Launched group 2 (ent 1.0/2.0, 12 runs) on GPU $GPU"

elif [ "$GROUP" = "3" ]; then
nohup bash -c "
for ENT in 5.0 10.0; do
for BETA in 0.0 0.25 0.5 1.0 5.0 10.0; do
  echo \"[\$(date +%H:%M)] ent=\$ENT beta=\$BETA\"
  $CMD algorithm.ENT_COEF=\$ENT algorithm.JA_BETA_MAX=\$BETA label=comm_sweep_ent\${ENT}_b\${BETA}_5s
done; done
echo \"[\$(date +%H:%M)] Group 3 Done\"
" > logs/comm_sweep_g3.log 2>&1 &
echo "Launched group 3 (ent 5.0/10.0, 12 runs) on GPU $GPU"

else
echo "Unknown group: $GROUP (use 1-3)"
exit 1
fi
