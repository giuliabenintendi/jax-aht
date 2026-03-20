#!/usr/bin/env bash
# Card game experiments.
# Usage: ./run_card_game.sh <gpu>
# Example: ./run_card_game.sh 0

GPU="${1:?Usage: ./run_card_game.sh <gpu>}"

mkdir -p logs

nohup bash -c "
echo \"[\$(date +%H:%M)] Starting image_ippo baseline (card-game)\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_image_ippo \
    task=card-game \
    algorithm=image_ippo/card-game \
    label=shuffled_cards

echo \"[\$(date +%H:%M)] Starting ja_ippo (card-game)\"
./run_gpu.sh $GPU marl.run \
    -cn base_config_ja_ippo \
    task=card-game \
    algorithm=ja_ippo/card-game \
    label=shuffled_cards

echo \"[\$(date +%H:%M)] All card-game runs complete\"
" > logs/card_game.log 2>&1 &

echo "Launched card-game experiments on GPU $GPU (check logs/card_game.log)"
