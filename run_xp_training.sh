#!/usr/bin/env bash
# Train JA-IPPO with 5 separate seeds for cross-play evaluation.
# Each seed is a separate wandb run (grouped for mean±std visualization).
#
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

SEEDS=(20374 48291 73615 91042 35768)
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

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    echo "=== JA-IPPO XP: $TASK seed $i ($SEED), beta=$BETA, gpu=$GPU ==="
    ./run_gpu.sh "$GPU" marl.run \
        -cn base_config_ja_ippo \
        task=$TASK_CFG \
        algorithm.NUM_SEEDS=1 \
        algorithm.TRAIN_SEED=$SEED \
        algorithm.JA_BETA_MAX=$BETA \
        label="${LABEL}/seed_${i}"
done

echo "All seeds complete for $TASK."
