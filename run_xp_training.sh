#!/usr/bin/env bash
# Train JA-IPPO with 5 seeds for cross-play evaluation.
# Usage: ./run_xp_training.sh <gpu_id> <task> <beta>
#
# Tasks: cramped_room, coord_ring, forced_coord, lbf-image-10food
#
# Examples:
#   ./run_xp_training.sh 1 cramped_room 0.1
#   ./run_xp_training.sh 5 lbf-image-10food 0.05

set -e

GPU="${1:?Usage: ./run_xp_training.sh <gpu_id> <task> <beta>}"
TASK="${2:?Usage: ./run_xp_training.sh <gpu_id> <task> <beta>}"
BETA="${3:?Usage: ./run_xp_training.sh <gpu_id> <task> <beta>}"
SEEDS=5
LABEL=xp_seeds

# Map short task names to hydra task configs
case "$TASK" in
    cramped_room|coord_ring|forced_coord)
        TASK_CFG="overcooked-v1/$TASK" ;;
    lbf-image-10food)
        TASK_CFG="lbf-image-10food" ;;
    *)
        echo "Unknown task: $TASK"
        echo "Available: cramped_room, coord_ring, forced_coord, lbf-image-10food"
        exit 1 ;;
esac

echo "=== JA-IPPO XP training: $TASK (beta=$BETA, seeds=$SEEDS, gpu=$GPU) ==="
./run_gpu.sh "$GPU" marl.run \
    -cn base_config_ja_ippo \
    task=$TASK_CFG \
    algorithm.NUM_SEEDS=$SEEDS \
    algorithm.JA_BETA_MAX=$BETA \
    label=$LABEL
