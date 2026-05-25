#!/usr/bin/env bash
# Per-chunk SP/XP eval for the over-training (Hu et al. style) plot.
# Runs the 3 headline conditions on a free GPU, sequentially.
#
# Usage:
#   bash run_over_training_eval.sh [GPU=1]
set -euo pipefail

GPU=${1:-1}
OUT=evaluation/card_game/over_training
mkdir -p "$OUT"

CKPT=/scratch/benintendi/jax-aht/checkpoints
RESULTS=/scratch/benintendi/jax-aht/results

ts() { date +%Y-%m-%d_%H:%M:%S; }

echo "[$(ts)] starting over-training eval on GPU $GPU"

# ---- 1. OP only (5M, ~25 chunks, eval every chunk) ----
echo "[$(ts)] --- OP only ---"
./run_gpu.sh "$GPU" evaluation.card_game.eval_over_training \
    --ckpt-root "$CKPT/splendid-salad-1474_card-game-op_ja_ippo_comm_rerun/op_only_48s_5M_s48_22052026" \
    --hydra-dir "$RESULTS/card-game-op/ja_ippo/comm_rerun/op_only_48s/2026-05-22_23-37-00" \
    --label "OP only" \
    --out-csv "$OUT/op_only.csv" \
    --every 1

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

echo "[$(ts)] all 3 conditions done."
echo "now run:  uv run python evaluation/card_game/plot_over_training.py"
