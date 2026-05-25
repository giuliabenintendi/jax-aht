#!/usr/bin/env bash
# Per-chunk SP/XP eval for the over-training (Hu et al. style) figure.
# Runs the 3 conditions on one free GPU, sequentially.
#
# Conditions:
#   1. NOT-OP baseline (replaces "OP only" — proper SP-collapse comparison)
#   2. OP + JA + shaping
#   3. OP + comm + shaping
#
# Usage:
#   bash run_over_training_eval.sh [GPU=1]
#
# IMPORTANT: For condition 1 (NOT-OP), you must first train the baseline via
# launch_not_op.sh and fill in the matching --ckpt-root / --hydra-dir paths
# below (search for FILL-IN comments).
set -euo pipefail

GPU=${1:-1}
OUT=evaluation/card_game/over_training
mkdir -p "$OUT"

CKPT=/scratch/benintendi/jax-aht/checkpoints
RESULTS=/scratch/benintendi/jax-aht/results

ts() { date +%Y-%m-%d_%H:%M:%S; }

echo "[$(ts)] starting over-training eval on GPU $GPU"

# ---- 1. NOT-OP baseline (5M, eval every chunk) ----
# After `launch_not_op.sh` finishes, locate its ckpt root and hydra dir:
#   find /scratch/benintendi/jax-aht/checkpoints -name chunk_scores.json -newer launch_not_op.sh | xargs -I{} dirname {}
#   find /scratch/benintendi/jax-aht/results/card-game -path "*not_op_48s_5M*" -name config.yaml | head
# Then fill the two paths below and uncomment the block.
echo "[$(ts)] --- NOT-OP ---"
# ./run_gpu.sh "$GPU" evaluation.card_game.eval_over_training \
#     --ckpt-root  "$CKPT/<RUN_NAME>/not_op_48s_5M_<TIMESTAMP>"   `# FILL-IN` \
#     --hydra-dir  "$RESULTS/card-game/ja_ippo/not_op_48s/<TIMESTAMP>"   `# FILL-IN` \
#     --label "Not OP" \
#     --out-csv "$OUT/not_op.csv" \
#     --every 1
echo "    (skipped — fill in NOT-OP ckpt-root + hydra-dir after launch_not_op.sh finishes)"

# ---- 2. OP + JA + shaping (15M, 77 chunks, eval every 4th) ----
echo "[$(ts)] --- OP + JA + shaping ---"
./run_gpu.sh "$GPU" evaluation.card_game.eval_over_training \
    --ckpt-root "$CKPT/likely-thunder-1466_card-game-op-delib-actions_ja_ippo_op_ja_shaped_48s_15M_s48_21052026" \
    --hydra-dir "$RESULTS/card-game-op-delib-actions/ja_ippo/op_ja_shaped_48s/2026-05-21_23-14-01" \
    --label "OP + JA + shaping" \
    --out-csv "$OUT/op_ja_shaping.csv" \
    --every 4

# ---- 3. OP + comm + shaping (5M, ~25 chunks, eval every chunk) ----
echo "[$(ts)] --- OP + comm + shaping ---"
./run_gpu.sh "$GPU" evaluation.card_game.eval_over_training \
    --ckpt-root "$CKPT/cosmic-thunder-1475_card-game-op_ja_ippo_comm_rerun/comm_full_48s_5M_s48_24052026" \
    --hydra-dir "$RESULTS/card-game-op/ja_ippo/comm_rerun/comm_full_48s/2026-05-24_13-47-51" \
    --label "OP + comm + shaping" \
    --out-csv "$OUT/op_comm_shaping.csv" \
    --every 1

echo "[$(ts)] OP conditions done."
echo "When NOT-OP training completes, fill in its paths above and re-run this script."
echo "Then plot:  uv run python evaluation/card_game/plot_over_training.py"
