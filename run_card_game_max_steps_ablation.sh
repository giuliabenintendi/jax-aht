#!/usr/bin/env bash
# Episode-length ablation. 8 vmapped seeds per max_steps value, 5M env steps
# each. JA architecture is loaded but all JA reward terms are off
# (JA_BETA_MAX=0.0, JA_CARD_ATTN stays at base default false), so this
# measures how deliberation length affects emergent coordination through the
# message channel alone — with the COMM_WARMUP curriculum delaying comm
# shaping to the halfway mark.
#
# Both task.ROLLOUT_LENGTH and task.ENV_KWARGS.max_steps are overridden so
# rollouts cover exactly one episode (no GAE bootstrap noise on partial
# episodes; matches the comment in marl/configs/task/card-game.yaml).
#
# Note: with NUM_SEEDS=8 vmapped and the default NUM_ENVS=1024 from
# card-game.yaml, you'll have ~8k parallel envs. If GPU 2 OOMs, append
# `algorithm.NUM_ENVS=512` to COMMON below.
#
# Usage: ./run_card_game_max_steps_ablation.sh <gpu>

GPU="${1:?Usage: ./run_card_game_max_steps_ablation.sh <gpu>}"
TIMESTEPS=5e6
NUM_SEEDS=8
MAX_STEPS_VALUES="2 4 12 16"

COMMON="algorithm.NUM_SEEDS=$NUM_SEEDS algorithm.TRAIN_SEED=42 \
algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
algorithm.COMMUNICATION=true algorithm.JA_BETA_MAX=0.0 \
algorithm.GAE_LAMBDA=0.95 algorithm.LR=5e-4 algorithm.ENT_COEF=0.01 algorithm.ANNEAL_LR=false \
algorithm.COMM_WARMUP_ENV_STEPS=2500000 algorithm.COMM_REWARD_START_SCALE=0.0 \
task.ENV_KWARGS.other_play_position_shuffle=true task.ENV_KWARGS.other_play_recolouring=true \
task.ENV_KWARGS.match_coef=0.1 task.ENV_KWARGS.stability_coef=0.05 task.ENV_KWARGS.follow_coef=0.5"

mkdir -p logs

nohup bash -c "
for MS in $MAX_STEPS_VALUES; do
    echo \"[\$(date +%H:%M)] Starting max_steps=\$MS\"
    ./run_gpu.sh $GPU marl.run \
        -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game \
        $COMMON \
        task.ENV_KWARGS.max_steps=\$MS task.ROLLOUT_LENGTH=\$MS \
        label=max_steps_ablation_\$MS
    echo \"[\$(date +%H:%M)] Finished max_steps=\$MS\"
done
echo \"[\$(date +%H:%M)] max_steps ablation complete\"
" > logs/card_game_max_steps_ablation.log 2>&1 &

echo "Launched max_steps ablation ($MAX_STEPS_VALUES) on GPU $GPU"
echo "  NUM_SEEDS=$NUM_SEEDS per cell, $TIMESTEPS env steps each"
echo "PID: $!  |  log: logs/card_game_max_steps_ablation.log"
