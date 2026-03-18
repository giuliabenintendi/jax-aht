#!/usr/bin/env bash
# Run greedy + stochastic eval on all high-entropy checkpoints,
# uploading results to the existing wandb runs.
# Split across 2 GPUs for speed.
#
# Usage:
#   ./run_greedy_eval.sh <gpu_a> <gpu_b>
# Example:
#   ./run_greedy_eval.sh 5 6

set -e

GPU_A="${1:?Usage: ./run_greedy_eval.sh <gpu_a> <gpu_b>}"
GPU_B="${2:?Usage: ./run_greedy_eval.sh <gpu_a> <gpu_b>}"

mkdir -p logs

# GPU_A: cramped_room (2) + forced_coord nojsdgae (1)
(
    ./run_gpu.sh "$GPU_A" evaluation.eval_greedy \
        --checkpoint results/overcooked-v1/cramped_room/ja_ippo/dual_cr_b025_ent045/2026-03-17_14-30-34/saved_train_run \
        --run-id zkk7lyk4 --num-episodes 256 --output-dir plots/

    ./run_gpu.sh "$GPU_A" evaluation.eval_greedy \
        --checkpoint results/overcooked-v1/cramped_room/ja_ippo/dual_cr_b025_jsdgae_ent045/2026-03-17_20-28-47/saved_train_run \
        --run-id ze45biub --num-episodes 256 --output-dir plots/

    ./run_gpu.sh "$GPU_A" evaluation.eval_greedy \
        --checkpoint results/overcooked-v1/forced_coord/ja_ippo/dual_fc_b05_ent040/2026-03-17_14-34-39/saved_train_run \
        --run-id 8eqsz9te --num-episodes 256 --output-dir plots/
) &

# GPU_B: coord_ring (2) + forced_coord jsdgae (1)
(
    ./run_gpu.sh "$GPU_B" evaluation.eval_greedy \
        --checkpoint results/overcooked-v1/coord_ring/ja_ippo/dual_cring_b0001_ent045/2026-03-17_14-33-58/saved_train_run \
        --run-id gp5w42fm --num-episodes 256 --output-dir plots/

    ./run_gpu.sh "$GPU_B" evaluation.eval_greedy \
        --checkpoint results/overcooked-v1/coord_ring/ja_ippo/dual_cring_b0001_jsdgae_ent045/2026-03-17_20-28-47/saved_train_run \
        --run-id t0chl2o4 --num-episodes 256 --output-dir plots/

    ./run_gpu.sh "$GPU_B" evaluation.eval_greedy \
        --checkpoint results/overcooked-v1/forced_coord/ja_ippo/dual_fc_b05_jsdgae_ent040/2026-03-17_20-28-47/saved_train_run \
        --run-id 73facnlw --num-episodes 256 --output-dir plots/
) &

wait
echo "[$(date +%H:%M)] All greedy evals complete"
