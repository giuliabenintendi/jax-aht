#!/usr/bin/env bash
# Static card game sweep v2 — remaining 53 runs.
# 500k steps, 8 seeds, warmup=50%.
# Usage: ./run_card_game_remaining.sh <gpu> <group>
#
# group 1: ent=0.1 remaining (9 runs) ~16h
# group 2: ent=0.2 remaining (4 runs) ~7h
# group 3: ent=0.01 (10 runs) ~18h
# group 4: ent=0.05 (10 runs) ~18h
# group 5: ent=0.8 (10 runs) ~18h
# group 6: ent=1.0 (10 runs) ~18h

GPU="${1:?Usage: ./run_card_game_remaining.sh <gpu> <group>}"
GROUP="${2:?Usage: ./run_card_game_remaining.sh <gpu> <group>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game algorithm.NUM_SEEDS=8 algorithm.TOTAL_TIMESTEPS=5e5 algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=250000"

if [ "$GROUP" = "1" ]; then
nohup bash -c "
echo \"[\$(date +%H:%M)] ent=0.1 beta=0.0 feed=false\"
$CMD algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=0.0 algorithm.FEED_OTHER_ATTN=false label=sweep2_ent0.1_b0.0_feedfalse_s8
for BETA in 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.1 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep2_ent0.1_b\${BETA}_feed\${FEED}_s8
done; done
echo \"[\$(date +%H:%M)] Group 1 Done\"
" > logs/remaining_g1.log 2>&1 &
echo "Launched group 1 (ent=0.1, 9 runs) on GPU $GPU"

elif [ "$GROUP" = "2" ]; then
nohup bash -c "
for BETA in 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.2 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.2 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep2_ent0.2_b\${BETA}_feed\${FEED}_s8
done; done
echo \"[\$(date +%H:%M)] Group 2 Done\"
" > logs/remaining_g2.log 2>&1 &
echo "Launched group 2 (ent=0.2 remaining, 4 runs) on GPU $GPU"

elif [ "$GROUP" = "3" ]; then
nohup bash -c "
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.01 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.01 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep2_ent0.01_b\${BETA}_feed\${FEED}_s8
done; done
echo \"[\$(date +%H:%M)] Group 3 Done\"
" > logs/remaining_g3.log 2>&1 &
echo "Launched group 3 (ent=0.01, 10 runs) on GPU $GPU"

elif [ "$GROUP" = "4" ]; then
nohup bash -c "
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.05 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.05 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep2_ent0.05_b\${BETA}_feed\${FEED}_s8
done; done
echo \"[\$(date +%H:%M)] Group 4 Done\"
" > logs/remaining_g4.log 2>&1 &
echo "Launched group 4 (ent=0.05, 10 runs) on GPU $GPU"

elif [ "$GROUP" = "5" ]; then
nohup bash -c "
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.8 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.8 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep2_ent0.8_b\${BETA}_feed\${FEED}_s8
done; done
echo \"[\$(date +%H:%M)] Group 5 Done\"
" > logs/remaining_g5.log 2>&1 &
echo "Launched group 5 (ent=0.8, 10 runs) on GPU $GPU"

elif [ "$GROUP" = "6" ]; then
nohup bash -c "
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=1.0 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=1.0 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep2_ent1.0_b\${BETA}_feed\${FEED}_s8
done; done
echo \"[\$(date +%H:%M)] Group 6 Done\"
" > logs/remaining_g6.log 2>&1 &
echo "Launched group 6 (ent=1.0, 10 runs) on GPU $GPU"

else
echo "Unknown group: $GROUP (use 1-6)"
exit 1
fi
