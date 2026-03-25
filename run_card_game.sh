#!/usr/bin/env bash
# Card game dynamic sweep. 500k steps, 6 seeds per run.
# Usage: ./run_card_game.sh <gpu> <group>
#   group 1: entropy sweep (GPU 1)
#   group 2: beta sweep (GPU 2)
#   group 3: warmup + extreme combos (GPU 4)
#   group 4: baselines + mid combos (GPU 5)

GPU="${1:?Usage: ./run_card_game.sh <gpu> <group>}"
GROUP="${2:?Usage: ./run_card_game.sh <gpu> <group>}"

mkdir -p logs

BASE="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game-dynamic algorithm=ja_ippo/card-game algorithm.NUM_SEEDS=6 algorithm.TOTAL_TIMESTEPS=5e5 algorithm.FEED_OTHER_ATTN=true algorithm.LR=8e-4"

if [ "$GROUP" = "1" ]; then
nohup bash -c "
echo '=== GROUP 1: Entropy sweep (beta=0.25, warmup=70%) ==='

echo \"[\$(date +%H:%M)] ent=0.1 beta=0.25\"
$BASE algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=0.25 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.1_b0.25_w70_s6

echo \"[\$(date +%H:%M)] ent=0.2 beta=0.25\"
$BASE algorithm.ENT_COEF=0.2 algorithm.JA_BETA_MAX=0.25 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.2_b0.25_w70_s6

echo \"[\$(date +%H:%M)] ent=0.3 beta=0.25\"
$BASE algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=0.25 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.3_b0.25_w70_s6

echo \"[\$(date +%H:%M)] ent=0.5 beta=0.25\"
$BASE algorithm.ENT_COEF=0.5 algorithm.JA_BETA_MAX=0.25 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.5_b0.25_w70_s6

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g1.log 2>&1 &
echo "Launched group 1 (entropy sweep) on GPU $GPU"

elif [ "$GROUP" = "2" ]; then
nohup bash -c "
echo '=== GROUP 2: Beta sweep (ent=0.1, warmup=70%) ==='

echo \"[\$(date +%H:%M)] beta=0.1\"
$BASE algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=0.1 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.1_b0.1_w70_s6

echo \"[\$(date +%H:%M)] beta=0.5\"
$BASE algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=0.5 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.1_b0.5_w70_s6

echo \"[\$(date +%H:%M)] beta=0.75\"
$BASE algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=0.75 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.1_b0.75_w70_s6

echo \"[\$(date +%H:%M)] beta=1.0\"
$BASE algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=1.0 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.1_b1.0_w70_s6

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g2.log 2>&1 &
echo "Launched group 2 (beta sweep) on GPU $GPU"

elif [ "$GROUP" = "3" ]; then
nohup bash -c "
echo '=== GROUP 3: Warmup sweep + extreme combos ==='

echo \"[\$(date +%H:%M)] warmup=50% (beta=0.5, ent=0.1)\"
$BASE algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=0.5 algorithm.JA_WARMUP_ENV_STEPS=250000 label=dyn_ent0.1_b0.5_w50_s6

echo \"[\$(date +%H:%M)] warmup=100% (beta=0.5, ent=0.1)\"
$BASE algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=0.5 algorithm.JA_WARMUP_ENV_STEPS=500000 label=dyn_ent0.1_b0.5_w100_s6

echo \"[\$(date +%H:%M)] ent=0.3 beta=0.75\"
$BASE algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=0.75 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.3_b0.75_w70_s6

echo \"[\$(date +%H:%M)] ent=0.5 beta=1.0\"
$BASE algorithm.ENT_COEF=0.5 algorithm.JA_BETA_MAX=1.0 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.5_b1.0_w70_s6

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g3.log 2>&1 &
echo "Launched group 3 (warmup + extreme) on GPU $GPU"

elif [ "$GROUP" = "4" ]; then
nohup bash -c "
echo '=== GROUP 4: Baselines + mid combos ==='

echo \"[\$(date +%H:%M)] baseline: beta=0 ent=0.01\"
$BASE algorithm.ENT_COEF=0.01 algorithm.JA_BETA_MAX=0.0 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_baseline_ent0.01_b0_s6

echo \"[\$(date +%H:%M)] baseline: beta=0 ent=0.1\"
$BASE algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=0.0 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_baseline_ent0.1_b0_s6

echo \"[\$(date +%H:%M)] ent=0.2 beta=0.5\"
$BASE algorithm.ENT_COEF=0.2 algorithm.JA_BETA_MAX=0.5 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.2_b0.5_w70_s6

echo \"[\$(date +%H:%M)] ent=0.3 beta=0.5\"
$BASE algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=0.5 algorithm.JA_WARMUP_ENV_STEPS=350000 label=dyn_ent0.3_b0.5_w70_s6

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_g4.log 2>&1 &
echo "Launched group 4 (baselines + mid) on GPU $GPU"

else
echo "Unknown group: $GROUP (use 1, 2, 3, or 4)"
exit 1
fi
