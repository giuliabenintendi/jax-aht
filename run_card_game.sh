#!/usr/bin/env bash
# Static card game sweep v2. 500k steps, 8 seeds, warmup=50%.
# 40 runs: 4 entropy x 5 beta x 2 feed_attn
# Usage: ./run_card_game.sh <gpu> <group>
#   group 1 (GPU 2): ent 0.1 — 10 runs
#   group 2 (GPU 3): ent 0.2 — 10 runs
#   group 3 (GPU 5): ent 0.3 — 10 runs
#   group 4 (GPU 6): ent 0.5 — 10 runs

GPU="${1:?Usage: ./run_card_game.sh <gpu> <group>}"
GROUP="${2:?Usage: ./run_card_game.sh <gpu> <group>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game algorithm.NUM_SEEDS=8 algorithm.TOTAL_TIMESTEPS=5e5 algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=250000"

if [ "$GROUP" = "1" ]; then
nohup bash -c "
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.1 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep2_ent0.1_b\${BETA}_feed\${FEED}_s8
done; done
echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g1.log 2>&1 &
echo "Launched group 1 (ent=0.1, 10 runs) on GPU $GPU"

elif [ "$GROUP" = "2" ]; then
nohup bash -c "
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.2 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.2 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep2_ent0.2_b\${BETA}_feed\${FEED}_s8
done; done
echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g2.log 2>&1 &
echo "Launched group 2 (ent=0.2, 10 runs) on GPU $GPU"

elif [ "$GROUP" = "3" ]; then
nohup bash -c "
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.3 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep2_ent0.3_b\${BETA}_feed\${FEED}_s8
done; done
echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g3.log 2>&1 &
echo "Launched group 3 (ent=0.3, 10 runs) on GPU $GPU"

elif [ "$GROUP" = "4" ]; then
nohup bash -c "
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.5 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.5 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep2_ent0.5_b\${BETA}_feed\${FEED}_s8
done; done
echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g4.log 2>&1 &
echo "Launched group 4 (ent=0.5, 10 runs) on GPU $GPU"

else
echo "Unknown group: $GROUP (use 1-4)"
exit 1
fi
