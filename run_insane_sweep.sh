#!/usr/bin/env bash
# Insane beta/entropy sweep on static card game.
# 500k steps, 5 seeds, feed=on.
# Usage: ./run_insane_sweep.sh <gpu> <group>
#   group 1: beta sweep (ent=0.1, beta=0/1/10/50/100/500)
#   group 2: entropy sweep (beta=0.25, ent=0.01/0.5/1/2/5/10)

GPU="${1:?Usage: ./run_insane_sweep.sh <gpu> <group>}"
GROUP="${2:?Usage: ./run_insane_sweep.sh <gpu> <group>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game algorithm.NUM_SEEDS=5 algorithm.TOTAL_TIMESTEPS=5e5 algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=250000 algorithm.FEED_OTHER_ATTN=true"

if [ "$GROUP" = "1" ]; then
nohup bash -c "
for BETA in 10.0 50.0 100.0 500.0; do
  echo \"[\$(date +%H:%M)] beta=\$BETA ent=0.1\"
  $CMD algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=\$BETA label=insane_b\${BETA}_ent0.1_5s
done
echo \"[\$(date +%H:%M)] Group 1 Done\"
" > logs/insane_g1.log 2>&1 &
echo "Launched group 1 (beta sweep) on GPU $GPU"

elif [ "$GROUP" = "2" ]; then
nohup bash -c "
for ENT in 2.0 5.0 10.0 20.0; do
  echo \"[\$(date +%H:%M)] beta=0.25 ent=\$ENT\"
  $CMD algorithm.ENT_COEF=\$ENT algorithm.JA_BETA_MAX=0.25 label=insane_b0.25_ent\${ENT}_5s
done
echo \"[\$(date +%H:%M)] Group 2 Done\"
" > logs/insane_g2.log 2>&1 &
echo "Launched group 2 (entropy sweep) on GPU $GPU"

else
echo "Unknown group: $GROUP (use 1-2)"
exit 1
fi
