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

# ---- 1. SP-baseline curve: OP-only checkpoints evaluated with OP wrappers off ----
# This exposes the agent's private canonical-color preference: SP=1.0 (same seed
# matches itself perfectly), XP=0.2 (different seeds picked different colors,
# chance match). This is the Hu et al. "Self-Play" line — reproduced via a
# different procedural route (OP-trained, eval w/o OP) because pure NOT-OP
# training hits a parameter-sharing symmetry trap without per-agent observation
# asymmetry. 5M training, ~25 chunks, every chunk, all 48 seeds (cheap).
echo "[$(ts)] --- OP only (drop-op eval) — SP baseline ---"
./run_gpu.sh "$GPU" evaluation.card_game.eval_over_training \
    --ckpt-root "$CKPT/splendid-salad-1474_card-game-op_ja_ippo_comm_rerun/op_only_48s_5M_s48_22052026" \
    --hydra-dir "$RESULTS/card-game-op/ja_ippo/comm_rerun/op_only_48s/2026-05-22_23-37-00" \
    --label "OP only (no-OP eval)" \
    --out-csv "$OUT/self_play.csv" \
    --every 1 \
    --drop-op \
    --select-n 10

# ---- 2. OP + JA + shaping (15M, 77 chunks, eval every 4th, 10-seed subset) ----
# --select-n 10 picks 10 seeds stratified by final-ckpt return rank so the
# subset's mean ≈ the 48-seed mean (preserves both center and shape).
echo "[$(ts)] --- OP + JA + shaping ---"
./run_gpu.sh "$GPU" evaluation.card_game.eval_over_training \
    --ckpt-root "$CKPT/likely-thunder-1466_card-game-op-delib-actions_ja_ippo_op_ja_shaped_48s_15M_s48_21052026" \
    --hydra-dir "$RESULTS/card-game-op-delib-actions/ja_ippo/op_ja_shaped_48s/2026-05-21_23-14-01" \
    --label "OP + JA + shaping" \
    --out-csv "$OUT/op_ja_shaping.csv" \
    --every 4 \
    --select-n 10

# ---- 3. OP + comm + shaping (5M, ~25 chunks, eval every chunk, 10-seed subset) ----
echo "[$(ts)] --- OP + comm + shaping ---"
./run_gpu.sh "$GPU" evaluation.card_game.eval_over_training \
    --ckpt-root "$CKPT/cosmic-thunder-1475_card-game-op_ja_ippo_comm_rerun/comm_full_48s_5M_s48_24052026" \
    --hydra-dir "$RESULTS/card-game-op/ja_ippo/comm_rerun/comm_full_48s/2026-05-24_13-47-51" \
    --label "OP + comm + shaping" \
    --out-csv "$OUT/op_comm_shaping.csv" \
    --every 1 \
    --select-n 10

echo "[$(ts)] all 3 conditions done."
echo "Then plot:  uv run python evaluation/card_game/plot_over_training.py"
