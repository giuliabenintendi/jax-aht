#!/usr/bin/env bash
# Compute attention stasis + object coverage for the 15 beta_sweep runs (5 seeds each).
# Splits across two GPUs for speed.
#
# Usage:
#   ./run_attention_eval.sh <gpu_a> <gpu_b>
# Example:
#   ./run_attention_eval.sh 5 6

set -e

GPU_A="${1:?Usage: ./run_attention_eval.sh <gpu_a> <gpu_b>}"
GPU_B="${2:?Usage: ./run_attention_eval.sh <gpu_a> <gpu_b>}"

# Cramped Room (5 seeds each)
CR_B0="results/overcooked-v1/cramped_room/ja_ippo/beta_sweep/2026-03-11_16-28-18/saved_train_run"
CR_B01="results/overcooked-v1/cramped_room/ja_ippo/beta_sweep/2026-03-12_02-19-02/saved_train_run"
CR_B025="results/overcooked-v1/cramped_room/ja_ippo/beta_sweep/2026-03-12_12-02-26/saved_train_run"
CR_B05="results/overcooked-v1/cramped_room/ja_ippo/beta_sweep/2026-03-12_22-28-29/saved_train_run"
CR_B1="results/overcooked-v1/cramped_room/ja_ippo/beta_sweep/2026-03-13_10-43-54/saved_train_run"

# Coord Ring (5 seeds each)
CRG_B0="results/overcooked-v1/coord_ring/ja_ippo/beta_sweep/2026-03-11_22-21-45/saved_train_run"
CRG_B01="results/overcooked-v1/coord_ring/ja_ippo/beta_sweep/2026-03-12_09-05-56/saved_train_run"
CRG_B025="results/overcooked-v1/coord_ring/ja_ippo/beta_sweep/2026-03-12_19-57-49/saved_train_run"
CRG_B05="results/overcooked-v1/coord_ring/ja_ippo/beta_sweep/2026-03-13_06-54-52/saved_train_run"
CRG_B1="results/overcooked-v1/coord_ring/ja_ippo/beta_sweep/2026-03-13_17-35-15/saved_train_run"

# Forced Coord (5 seeds each)
FC_B0="results/overcooked-v1/forced_coord/ja_ippo/beta_sweep/2026-03-11_22-21-40/saved_train_run"
FC_B01="results/overcooked-v1/forced_coord/ja_ippo/beta_sweep/2026-03-12_08-59-05/saved_train_run"
FC_B025="results/overcooked-v1/forced_coord/ja_ippo/beta_sweep/2026-03-12_19-46-01/saved_train_run"
FC_B05="results/overcooked-v1/forced_coord/ja_ippo/beta_sweep/2026-03-13_06-32-41/saved_train_run"
FC_B1="results/overcooked-v1/forced_coord/ja_ippo/beta_sweep/2026-03-13_17-17-24/saved_train_run"

# GPU_A: cramped_room (5) + forced_coord first 3
(
    for CKPT in "$CR_B0" "$CR_B01" "$CR_B025" "$CR_B05" "$CR_B1" "$FC_B0" "$FC_B01" "$FC_B025"; do
        echo "[$(date +%H:%M)] GPU_A: $CKPT"
        ./run_gpu.sh "$GPU_A" evaluation.compute_attention_metrics --checkpoint "$CKPT"
    done
) &

# GPU_B: coord_ring (5) + forced_coord last 2
(
    for CKPT in "$CRG_B0" "$CRG_B01" "$CRG_B025" "$CRG_B05" "$CRG_B1" "$FC_B05" "$FC_B1"; do
        echo "[$(date +%H:%M)] GPU_B: $CKPT"
        ./run_gpu.sh "$GPU_B" evaluation.compute_attention_metrics --checkpoint "$CKPT"
    done
) &

wait
echo "[$(date +%H:%M)] All attention evals complete"
