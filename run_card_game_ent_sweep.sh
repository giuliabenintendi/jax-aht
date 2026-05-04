#!/usr/bin/env bash
# Card game ENT_COEF sweep on a single seed.
# Single-seed (TRAIN_SEED=43) so curve differences come purely from ENT_COEF.
# Usage: ./run_card_game_ent_sweep.sh <gpu>

GPU="${1:?Usage: ./run_card_game_ent_sweep.sh <gpu>}"
SEED=43
TIMESTEPS=5e6
ENT_VALUES="0.05 0.1 0.2"

COMMON="algorithm.NUM_SEEDS=1 algorithm.TRAIN_SEED=$SEED \
algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
algorithm.COMMUNICATION=true algorithm.JA_BETA_MAX=0.0 \
algorithm.GAE_LAMBDA=0.95 algorithm.LR=5e-4 algorithm.ANNEAL_LR=false \
algorithm.COMM_WARMUP_ENV_STEPS=0 \
task.ENV_KWARGS.other_play_position_shuffle=true task.ENV_KWARGS.other_play_recolouring=true \
task.ENV_KWARGS.match_coef=0.1 task.ENV_KWARGS.stability_coef=0.05 task.ENV_KWARGS.follow_coef=0.5"

mkdir -p logs

nohup bash -c "
for ENT in $ENT_VALUES; do
    echo \"[\$(date +%H:%M)] Starting ENT_COEF=\$ENT\"
    ./run_gpu.sh $GPU marl.run \
        -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
        $COMMON algorithm.ENT_COEF=\$ENT \
        label=ent_sweep_seed${SEED}
    echo \"[\$(date +%H:%M)] Finished ENT_COEF=\$ENT\"
done
echo \"[\$(date +%H:%M)] ENT sweep complete\"
" > logs/card_game_ent_sweep.log 2>&1 &

echo "Launched ENT_COEF sweep ($ENT_VALUES) seed=$SEED on GPU $GPU"
echo "PID: $!  |  log: logs/card_game_ent_sweep.log"
