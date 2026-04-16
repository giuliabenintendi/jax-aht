#!/usr/bin/env bash
# LR sweep: 6 configs across 2 GPUs, 1 seed each, 5M steps, sequential per GPU.
# GPU 3: MB=16 EP=4 (default) with LR=1e-3, 7e-4, 4e-4
# GPU 6: MB=8  EP=6            with LR=1e-3, 7e-4, 4e-4
# All use: NUM_ENVS=64, GAE=0.95, comm=0.5, follow=2.0, stability=0.02, no LR anneal

COMMON="marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
  algorithm.COMMUNICATION=true algorithm.JA_BETA_MAX=0.0 \
  algorithm.NUM_SEEDS=1 algorithm.NUM_ENVS=64 algorithm.GAE_LAMBDA=0.95 \
  algorithm.TOTAL_TIMESTEPS=5000000 algorithm.COMM_WARMUP_ENV_STEPS=0 \
  task.ENV_KWARGS.other_play_position_shuffle=true \
  task.ENV_KWARGS.other_play_recolouring=true \
  task.ENV_KWARGS.comm_reward_coef=0.5 \
  task.ENV_KWARGS.comm_follow_bonus=2.0 \
  task.ENV_KWARGS.comm_stability_bonus=0.02"

# GPU 3: default minibatches (MB=16, EP=4)
nohup bash -c "
  echo '[GPU3] Starting LR=1e-3 MB=16 EP=4'
  ./run_gpu.sh 3 $COMMON algorithm.LR=1e-3
  echo '[GPU3] Starting LR=7e-4 MB=16 EP=4'
  ./run_gpu.sh 3 $COMMON algorithm.LR=7e-4
  echo '[GPU3] Starting LR=4e-4 MB=16 EP=4'
  ./run_gpu.sh 3 $COMMON algorithm.LR=4e-4
  echo '[GPU3] All done'
" > /tmp/gpu3_lr_sweep.log 2>&1 &

# GPU 6: MB=8, EP=6
nohup bash -c "
  echo '[GPU6] Starting LR=1e-3 MB=8 EP=6'
  ./run_gpu.sh 6 $COMMON algorithm.LR=1e-3 algorithm.NUM_MINIBATCHES=8 algorithm.UPDATE_EPOCHS=6
  echo '[GPU6] Starting LR=7e-4 MB=8 EP=6'
  ./run_gpu.sh 6 $COMMON algorithm.LR=7e-4 algorithm.NUM_MINIBATCHES=8 algorithm.UPDATE_EPOCHS=6
  echo '[GPU6] Starting LR=4e-4 MB=8 EP=6'
  ./run_gpu.sh 6 $COMMON algorithm.LR=4e-4 algorithm.NUM_MINIBATCHES=8 algorithm.UPDATE_EPOCHS=6
  echo '[GPU6] All done'
" > /tmp/gpu6_lr_sweep.log 2>&1 &

echo "Sweep launched on GPU 3 and 6. ~6 hours total."
echo "Logs: /tmp/gpu3_lr_sweep.log, /tmp/gpu6_lr_sweep.log"
