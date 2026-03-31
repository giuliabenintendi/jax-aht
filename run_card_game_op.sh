#!/usr/bin/env bash
# Other-Play card game: OP with and without communication.
# ENT=0.01 (default), BETA=0 (no JA reward), 5 seeds, 1M steps.
# Usage: ./run_card_game_op.sh <gpu>

GPU="${1:?Usage: ./run_card_game_op.sh <gpu>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game algorithm=ja_ippo/card-game algorithm.TOTAL_TIMESTEPS=1e6 algorithm.JA_BETA_MAX=0.0 algorithm.FEED_OTHER_ATTN=false task.ENV_KWARGS.other_play_position_shuffle=true task.ENV_KWARGS.other_play_recolouring=true"

nohup bash -c "
echo \"[\$(date +%H:%M)] OP without communication\"
$CMD algorithm.COMMUNICATION=false label=op_no_comm_s5

echo \"[\$(date +%H:%M)] OP with communication\"
$CMD algorithm.COMMUNICATION=true task.ENV_KWARGS.communication=true label=op_comm_s5

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_op.log 2>&1 &
echo "Launched OP sweep (2 runs) on GPU $GPU"
