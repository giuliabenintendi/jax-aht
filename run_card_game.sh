#!/usr/bin/env bash
# Static card game sweep. 250k steps, 6 seeds, warmup=70%.
# 80 runs: 8 entropy x 5 beta x 2 feed_attn
# Usage: ./run_card_game.sh <gpu> <group>

GPU="${1:?Usage: ./run_card_game.sh <gpu> <group>}"
GROUP="${2:?Usage: ./run_card_game.sh <gpu> <group>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game algorithm.NUM_SEEDS=6 algorithm.TOTAL_TIMESTEPS=2.5e5 algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=175000"

if [ "$GROUP" = "1" ]; then
nohup bash -c "
for ENT in 0.01 0.05; do
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=\$ENT beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=\$ENT algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent\${ENT}_b\${BETA}_feed\${FEED}_s6
done; done; done
echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g1.log 2>&1 &
echo "Launched group 1 (ent=0.01,0.05) on GPU $GPU"

elif [ "$GROUP" = "2" ]; then
nohup bash -c "
for ENT in 0.1 0.2; do
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=\$ENT beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=\$ENT algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent\${ENT}_b\${BETA}_feed\${FEED}_s6
done; done; done
echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g2.log 2>&1 &
echo "Launched group 2 (ent=0.1,0.2) on GPU $GPU"

elif [ "$GROUP" = "3" ]; then
nohup bash -c "
for ENT in 0.3 0.5; do
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=\$ENT beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=\$ENT algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent\${ENT}_b\${BETA}_feed\${FEED}_s6
done; done; done
echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g3.log 2>&1 &
echo "Launched group 3 (ent=0.3,0.5) on GPU $GPU"

elif [ "$GROUP" = "4" ]; then
nohup bash -c "
for ENT in 0.8 1.0; do
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=\$ENT beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=\$ENT algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent\${ENT}_b\${BETA}_feed\${FEED}_s6
done; done; done
echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g4.log 2>&1 &
echo "Launched group 4 (ent=0.8,1.0) on GPU $GPU"

else
echo "Unknown group: $GROUP (use 1, 2, 3, or 4)"
exit 1
fi
