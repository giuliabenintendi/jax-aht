#!/usr/bin/env bash
# Run XP eval on all multi-seed runs that don't have xp_results yet.
# All sequential on one GPU.
#
# Usage:
#   ./run_missing_xp.sh <gpu>
# Example:
#   ./run_missing_xp.sh 4

GPU="${1:?Usage: ./run_missing_xp.sh <gpu>}"

mkdir -p logs

nohup bash -c "
echo \"[\$(date +%H:%M)] CR dual b0.25 ent0.01 3s\"
./run_gpu.sh $GPU evaluation.run_xp_seeds \
    --checkpoint results/overcooked-v1/cramped_room/ja_ippo/dual_cr_b025/2026-03-16_21-39-15/saved_train_run

echo \"[\$(date +%H:%M)] CR dual b0 ent0.45 3s\"
./run_gpu.sh $GPU evaluation.run_xp_seeds \
    --checkpoint results/overcooked-v1/cramped_room/ja_ippo/dual_cr_b0_ent045_3s/2026-03-19_08-05-28/saved_train_run

echo \"[\$(date +%H:%M)] CR dual b0.25 ent0.45 3s\"
./run_gpu.sh $GPU evaluation.run_xp_seeds \
    --checkpoint results/overcooked-v1/cramped_room/ja_ippo/dual_cr_b025_ent045_3s/2026-03-19_08-05-28/saved_train_run

echo \"[\$(date +%H:%M)] CRing dual b0 ent0.45 5s\"
./run_gpu.sh $GPU evaluation.run_xp_seeds \
    --checkpoint results/overcooked-v1/coord_ring/ja_ippo/dual_cring_b0_ent045_5s/2026-03-19_22-46-42/saved_train_run

echo \"[\$(date +%H:%M)] FC dual b0.5 ent0.01 3s\"
./run_gpu.sh $GPU evaluation.run_xp_seeds \
    --checkpoint results/overcooked-v1/forced_coord/ja_ippo/dual_fc_b05/2026-03-17_10-45-09/saved_train_run

echo \"[\$(date +%H:%M)] All missing XP evals complete\"
" > logs/missing_xp.log 2>&1 &

echo "Launched 5 sequential XP evals on GPU $GPU (check logs/missing_xp.log)"
