#!/usr/bin/env bash
# Flip card game experiments. 500k steps, 5 seeds, warmup=50%.
# 8 runs sequential ≈ 8 hours on 1 GPU.
# Usage: ./run_flip_game.sh <gpu>

GPU="${1:?Usage: ./run_flip_game.sh <gpu>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game-flip algorithm=ja_ippo/card-game algorithm.NUM_SEEDS=5 algorithm.TOTAL_TIMESTEPS=5e5 algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=250000"

nohup bash -c "
echo \"[\$(date +%H:%M)] 1/8 ent=0.1 beta=0 feed=off (baseline)\"
$CMD algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=0.0 algorithm.FEED_OTHER_ATTN=false label=flip_ent0.1_b0_5s

echo \"[\$(date +%H:%M)] 2/8 ent=0.3 beta=0 feed=off (baseline)\"
$CMD algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=0.0 algorithm.FEED_OTHER_ATTN=false label=flip_ent0.3_b0_5s

echo \"[\$(date +%H:%M)] 3/8 ent=0.5 beta=0 feed=off (baseline)\"
$CMD algorithm.ENT_COEF=0.5 algorithm.JA_BETA_MAX=0.0 algorithm.FEED_OTHER_ATTN=false label=flip_ent0.5_b0_5s

echo \"[\$(date +%H:%M)] 4/8 ent=0.1 beta=0.25 feed=on\"
$CMD algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=0.25 algorithm.FEED_OTHER_ATTN=true label=flip_ent0.1_b0.25_feed_5s

echo \"[\$(date +%H:%M)] 5/8 ent=0.3 beta=0.5 feed=on\"
$CMD algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=0.5 algorithm.FEED_OTHER_ATTN=true label=flip_ent0.3_b0.5_feed_5s

echo \"[\$(date +%H:%M)] 6/8 ent=0.5 beta=0.5 feed=on\"
$CMD algorithm.ENT_COEF=0.5 algorithm.JA_BETA_MAX=0.5 algorithm.FEED_OTHER_ATTN=true label=flip_ent0.5_b0.5_feed_5s

echo \"[\$(date +%H:%M)] 7/8 ent=0.1 beta=1.0 feed=on\"
$CMD algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=1.0 algorithm.FEED_OTHER_ATTN=true label=flip_ent0.1_b1.0_feed_5s

echo \"[\$(date +%H:%M)] 8/8 ent=0.3 beta=1.0 feed=on\"
$CMD algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=1.0 algorithm.FEED_OTHER_ATTN=true label=flip_ent0.3_b1.0_feed_5s

echo \"[\$(date +%H:%M)] All done\"
" > logs/flip_game.log 2>&1 &

echo "Launched 8 flip game runs on GPU $GPU (check logs/flip_game.log)"
