#!/usr/bin/env bash
# Other-play card game runs. 1M steps.
# Usage: ./run_other_play.sh <gpu>

GPU="${1:?Usage: ./run_other_play.sh <gpu>}"

mkdir -p logs

nohup bash -c "
# 1. ja_ippo shared params, no comm, no feed_attn
echo \"[\$(date +%H:%M)] OP ja_ippo no-comm\"
./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
  algorithm.NUM_SEEDS=5 \
  algorithm.TOTAL_TIMESTEPS=1e6 \
  algorithm.JA_BETA_MAX=0.0 \
  algorithm.FEED_OTHER_ATTN=false \
  algorithm.COMMUNICATION=false \
  task.ENV_KWARGS.other_play_position_shuffle=true \
  task.ENV_KWARGS.other_play_recolouring=true \
  label=op_no_comm_s5

# 2. ja_ippo_no_share, comm + feed_attn
echo \"[\$(date +%H:%M)] OP ja_ippo_no_share comm\"
./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo_no_share/card-game \
  algorithm.NUM_SEEDS=5 \
  algorithm.TOTAL_TIMESTEPS=1e6 \
  algorithm.LR=8e-4 \
  algorithm.JA_WARMUP_ENV_STEPS=500000 \
  algorithm.COMMUNICATION=true \
  algorithm.FEED_OTHER_ATTN=true \
  algorithm.JA_BETA_MAX=0.0 \
  algorithm.ENT_COEF=0.1 \
  task.ENV_KWARGS.other_play_position_shuffle=true \
  task.ENV_KWARGS.other_play_recolouring=true \
  label=op_no_share_comm_b0_ent0.1_1M_5s

echo \"[\$(date +%H:%M)] Done\"
" > logs/other_play.log 2>&1 &

echo "Launched 2 other-play runs on GPU $GPU (check logs/other_play.log)"
