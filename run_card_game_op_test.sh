#!/usr/bin/env bash
# OP-test diagnostic: train the 4-white-1-red card game with and without
# position shuffle, 3 seeds each, 500k steps. Expectation:
#   no-OP  → self-play return converges near 1.0 (position-based white pick);
#            cross-play collapses (seeds learn different "pick position N").
#   OP     → self-play and cross-play both converge near 0.9 (pick red,
#            the only OP-invariant signal).
# Usage: ./run_card_game_op_test.sh <gpu>

GPU="${1:?Usage: ./run_card_game_op_test.sh <gpu>}"

mkdir -p logs

CMD="./run_gpu.sh $GPU marl.run -cn base_config_ja_ippo task=card-game-op-test algorithm=ja_ippo/card-game-op-test algorithm.NUM_SEEDS=3 algorithm.TOTAL_TIMESTEPS=5e5"

nohup bash -c "
echo \"[\$(date +%H:%M)] baseline: no OP\"
$CMD task.ENV_KWARGS.op_position_shuffle=false label=op_test_noop_s3

echo \"[\$(date +%H:%M)] OP: position shuffle enabled\"
$CMD task.ENV_KWARGS.op_position_shuffle=true  label=op_test_op_s3

echo \"[\$(date +%H:%M)] Done\"
" > logs/card_game_op_test.log 2>&1 &

echo "Launched OP-test diagnostic (2 runs x 3 seeds) on GPU $GPU"
echo "  log: logs/card_game_op_test.log"
