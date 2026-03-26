#!/usr/bin/env bash
# Static card game sweep — remaining 59 runs.
# 250k steps, 6 seeds, warmup=70%.
# Usage: ./run_card_game.sh <gpu> <group>
#   group 1: 30 runs (ent 0.01, 0.05, 0.1 remaining, 0.2)
#   group 2: 29 runs (ent 0.3 remaining, 0.5, 0.8, 1.0)

GPU="${1:?Usage: ./run_card_game.sh <gpu> <group>}"
GROUP="${2:?Usage: ./run_card_game.sh <gpu> <group>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game algorithm.NUM_SEEDS=6 algorithm.TOTAL_TIMESTEPS=2.5e5 algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=175000"

if [ "$GROUP" = "1" ]; then
nohup bash -c "
# ent=0.01: all 10 runs
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.01 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.01 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent0.01_b\${BETA}_feed\${FEED}_s6
done; done

# ent=0.05: all 10 runs
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.05 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.05 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent0.05_b\${BETA}_feed\${FEED}_s6
done; done

# ent=0.1: missing 6 (b=0.25,0.5,1.0 x feed on/off)
for BETA in 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.1 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent0.1_b\${BETA}_feed\${FEED}_s6
done; done

# ent=0.2: all 10 runs
for BETA in 0.0 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.2 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.2 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent0.2_b\${BETA}_feed\${FEED}_s6
done; done

echo \"[\$(date +%H:%M)] Group 1 Done\"
" > logs/card_game_g1.log 2>&1 &
echo "Launched group 1 (30 runs: ent 0.01,0.05,0.1 remaining,0.2) on GPU $GPU"

elif [ "$GROUP" = "2" ]; then
nohup bash -c "
# ent=0.3: missing 1 (b=0.25 feed=off)
echo \"[\$(date +%H:%M)] ent=0.3 beta=0.25 feed=false\"
$CMD algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=0.25 algorithm.FEED_OTHER_ATTN=false label=sweep1_ent0.3_b0.25_feedfalse_s6

# ent=0.5: missing 8 (b=0.1,0.25,0.5,1.0 x feed on/off)
for BETA in 0.1 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.5 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.5 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent0.5_b\${BETA}_feed\${FEED}_s6
done; done

# ent=0.8: missing 7 (b=0.1 off, b=0.25 on/off, b=0.5 on/off, b=1.0 on/off)
echo \"[\$(date +%H:%M)] ent=0.8 beta=0.1 feed=false\"
$CMD algorithm.ENT_COEF=0.8 algorithm.JA_BETA_MAX=0.1 algorithm.FEED_OTHER_ATTN=false label=sweep1_ent0.8_b0.1_feedfalse_s6
for BETA in 0.25 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=0.8 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=0.8 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent0.8_b\${BETA}_feed\${FEED}_s6
done; done

# ent=1.0: missing 7 (b=0.0 on/off, b=0.25 off, b=0.5 on/off, b=1.0 on/off)
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=1.0 beta=0.0 feed=\$FEED\"
  $CMD algorithm.ENT_COEF=1.0 algorithm.JA_BETA_MAX=0.0 algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent1.0_b0.0_feed\${FEED}_s6
done
echo \"[\$(date +%H:%M)] ent=1.0 beta=0.25 feed=false\"
$CMD algorithm.ENT_COEF=1.0 algorithm.JA_BETA_MAX=0.25 algorithm.FEED_OTHER_ATTN=false label=sweep1_ent1.0_b0.25_feedfalse_s6
for BETA in 0.5 1.0; do
for FEED in true false; do
  echo \"[\$(date +%H:%M)] ent=1.0 beta=\$BETA feed=\$FEED\"
  $CMD algorithm.ENT_COEF=1.0 algorithm.JA_BETA_MAX=\$BETA algorithm.FEED_OTHER_ATTN=\$FEED label=sweep1_ent1.0_b\${BETA}_feed\${FEED}_s6
done; done

echo \"[\$(date +%H:%M)] Group 2 Done\"
" > logs/card_game_g2.log 2>&1 &
echo "Launched group 2 (29 runs: ent 0.3 remaining,0.5,0.8,1.0) on GPU $GPU"

else
echo "Unknown group: $GROUP (use 1 or 2)"
exit 1
fi
