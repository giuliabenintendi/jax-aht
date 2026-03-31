#!/usr/bin/env bash
# No-share + communication, beta=0, 1M steps, 5 seeds.
# Varying entropy. No JA pressure — pure communication.
# Usage: ./run_no_share_comm.sh <gpu> <group>
#   group 1: ent 0.1, 0.3 = 2 runs (0.01 already done)
#   group 2: ent 0.5, 1.0 = 2 runs (already done)

GPU="${1:?Usage: ./run_no_share_comm.sh <gpu> <group>}"
GROUP="${2:?Usage: ./run_no_share_comm.sh <gpu> <group>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo_no_share/card-game algorithm.NUM_SEEDS=5 algorithm.TOTAL_TIMESTEPS=1e6 algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=500000 algorithm.COMMUNICATION=true algorithm.FEED_OTHER_ATTN=true algorithm.JA_BETA_MAX=0.0"

if [ "$GROUP" = "1" ]; then
nohup bash -c "
for ENT in 0.1 0.3; do
  echo \"[\$(date +%H:%M)] no-share comm b0 ent=\$ENT\"
  $CMD algorithm.ENT_COEF=\$ENT label=no_share_comm_b0_ent\${ENT}_1M_5s
done
echo \"[\$(date +%H:%M)] Group 1 Done\"
" > logs/no_share_comm_g1.log 2>&1 &
echo "Launched group 1 (ent 0.1/0.3, 2 runs) on GPU $GPU"

elif [ "$GROUP" = "2" ]; then
nohup bash -c "
for ENT in 0.5 1.0; do
  echo \"[\$(date +%H:%M)] no-share comm b0 ent=\$ENT\"
  $CMD algorithm.ENT_COEF=\$ENT label=no_share_comm_b0_ent\${ENT}_1M_5s
done
echo \"[\$(date +%H:%M)] Group 2 Done\"
" > logs/no_share_comm_g2.log 2>&1 &
echo "Launched group 2 (ent 0.5/1.0, 2 runs) on GPU $GPU"

else
echo "Unknown group: $GROUP (use 1-2)"
exit 1
fi
