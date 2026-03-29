#!/usr/bin/env bash
# Flip card game experiments. 500k steps, 5 seeds, warmup=50%.
# All runs: feed_attn=true. 6 runs ≈ 6 hours.
# Usage: ./run_flip_game.sh <gpu>

GPU="${1:?Usage: ./run_flip_game.sh <gpu>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game-flip algorithm=ja_ippo/card-game algorithm.NUM_SEEDS=5 algorithm.TOTAL_TIMESTEPS=5e5 algorithm.LR=8e-4 algorithm.JA_WARMUP_ENV_STEPS=250000 algorithm.FEED_OTHER_ATTN=true"

nohup bash -c "
echo \"[\$(date +%H:%M)] 1/6 ent=0.3 beta=0 (baseline)\"
$CMD algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=0.0 label=flip_ent0.3_b0_feed_5s

echo \"[\$(date +%H:%M)] 2/6 ent=0.5 beta=0 (baseline)\"
$CMD algorithm.ENT_COEF=0.5 algorithm.JA_BETA_MAX=0.0 label=flip_ent0.5_b0_feed_5s

echo \"[\$(date +%H:%M)] 3/6 ent=0.3 beta=0.5\"
$CMD algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=0.5 label=flip_ent0.3_b0.5_feed_5s

echo \"[\$(date +%H:%M)] 4/6 ent=0.5 beta=0.5\"
$CMD algorithm.ENT_COEF=0.5 algorithm.JA_BETA_MAX=0.5 label=flip_ent0.5_b0.5_feed_5s

echo \"[\$(date +%H:%M)] 5/6 ent=0.1 beta=1.0\"
$CMD algorithm.ENT_COEF=0.1 algorithm.JA_BETA_MAX=1.0 label=flip_ent0.1_b1.0_feed_5s

echo \"[\$(date +%H:%M)] 6/6 ent=0.3 beta=1.0\"
$CMD algorithm.ENT_COEF=0.3 algorithm.JA_BETA_MAX=1.0 label=flip_ent0.3_b1.0_feed_5s

echo \"[\$(date +%H:%M)] All done\"
" > logs/flip_game.log 2>&1 &

echo "Launched 6 flip game runs on GPU $GPU (check logs/flip_game.log)"
