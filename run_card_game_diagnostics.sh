#!/usr/bin/env bash
# Two single-seed (43) diagnostic runs to test fixes for the trivial-protocol collapse:
#   1) Reduced batch dimension (NUM_ENVS 1024 -> 128) -> noisier gradient, more exploration.
#   2) Comm reward annealing (start scale 0.0, full at 50% of training) -> env reward
#      drives initial learning before shaping ramps in.
# Both are 5M timesteps each, run sequentially.
# Usage: ./run_card_game_diagnostics.sh <gpu>

GPU="${1:?Usage: ./run_card_game_diagnostics.sh <gpu>}"
SEED=43
TIMESTEPS=5e6
HALF_TIMESTEPS=2500000

COMMON="algorithm.NUM_SEEDS=1 algorithm.TRAIN_SEED=$SEED \
algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
algorithm.COMMUNICATION=true algorithm.JA_BETA_MAX=0.0 \
algorithm.GAE_LAMBDA=0.95 algorithm.LR=5e-4 algorithm.ENT_COEF=0.01 algorithm.ANNEAL_LR=false \
task.ENV_KWARGS.other_play_position_shuffle=true task.ENV_KWARGS.other_play_recolouring=true \
task.ENV_KWARGS.match_coef=0.1 task.ENV_KWARGS.stability_coef=0.05 task.ENV_KWARGS.follow_coef=0.5"

mkdir -p logs

nohup bash -c "

echo \"[\$(date +%H:%M)] === Run 1/2: reduced batch (NUM_ENVS=128) ===\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    $COMMON \
    algorithm.NUM_ENVS=128 \
    algorithm.COMM_WARMUP_ENV_STEPS=0 \
    label=diag_small_batch_seed${SEED}
echo \"[\$(date +%H:%M)] === Run 1/2 done ===\"

echo \"[\$(date +%H:%M)] === Run 2/2: comm warmup (linear 0 to 1 at 50%) ===\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
    $COMMON \
    algorithm.COMM_WARMUP_ENV_STEPS=$HALF_TIMESTEPS \
    algorithm.COMM_REWARD_START_SCALE=0.0 \
    label=diag_comm_warmup_seed${SEED}
echo \"[\$(date +%H:%M)] === Run 2/2 done ===\"

echo \"[\$(date +%H:%M)] === All diagnostics complete ===\"
" > logs/card_game_diagnostics.log 2>&1 &

echo "Launched diagnostics seed=$SEED on GPU $GPU"
echo "PID: $!  |  log: logs/card_game_diagnostics.log"
