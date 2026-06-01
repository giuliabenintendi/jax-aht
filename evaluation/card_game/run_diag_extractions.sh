#!/usr/bin/env bash
#
# Extract Dec-POMDP diagnostics trajectories for the 5 card-game conditions.
# Runs on the GPU box; each condition writes evaluation/card_game/diag_data/<cond>.npz.
#
# Usage:
#   bash evaluation/card_game/run_diag_extractions.sh [GPU_ID] [NUM_EPISODES]
#
# Env overrides:
#   RESULTS       results root (default /scratch/benintendi/jax-aht/results)
#   MAX_SEEDS     cap SP seeds per condition (e.g. 12 for a fast first pass)
#   MAX_XP_PAIRS  cap XP seed pairs (default 6 in the extractor)
#
# Only ja_shape's path is verified; the other four are inferred from each run's
# W&B group. The script resolves every path up front and aborts if any is
# missing or ambiguous, so fix a wrong path before any GPU time is spent.
set -uo pipefail
shopt -s nullglob

GPU="${1:-5}"
EPISODES="${2:-1000}"
RESULTS="${RESULTS:-/scratch/benintendi/jax-aht/results}"

EXTRA=()
[[ -n "${MAX_SEEDS:-}" ]] && EXTRA+=(--max-seeds "$MAX_SEEDS")
[[ -n "${MAX_XP_PAIRS:-}" ]] && EXTRA+=(--max-xp-pairs "$MAX_XP_PAIRS")

# condition -> saved_train_run path. A '*' resolves the timestamp dir; ja_shape
# is pinned to the reconstructed checkpoint (it has two timestamp dirs, only one
# of which is the right run).
declare -A CKPT=(
  [op_only]="$RESULTS/card-game-op/ja_ippo/comm_rerun/op_only_48s/*/saved_train_run"
  [ja_noshape]="$RESULTS/card-game-op-delib-actions/ja_ippo/op_ja_48s/*/saved_train_run"
  [ja_shape]="$RESULTS/card-game-op-delib-actions/ja_ippo/op_ja_shaped_48s/2026-05-21_23-14-01/saved_train_run"
  [comm_noshape]="$RESULTS/card-game-op/ja_ippo/op_comm_noshape_5M_48s_g1/*/saved_train_run"
  [comm_shape]="$RESULTS/card-game-op/ja_ippo/comm_rerun/comm_full_48s/*/saved_train_run"
)
ORDER=(op_only ja_noshape ja_shape comm_noshape comm_shape)

declare -A RESOLVED
missing=0
echo "Resolving checkpoints under $RESULTS ..."
for cond in "${ORDER[@]}"; do
  hits=( ${CKPT[$cond]} )
  if (( ${#hits[@]} == 0 )); then
    echo "  ✘ $cond: NO match for ${CKPT[$cond]}"
    missing=1
  elif (( ${#hits[@]} > 1 )); then
    chosen="$(printf '%s\n' "${hits[@]}" | sort | tail -1)"
    echo "  ‼ $cond: ${#hits[@]} matches, using latest: $chosen"
    RESOLVED[$cond]="$chosen"
  else
    echo "  ✔ $cond: ${hits[0]}"
    RESOLVED[$cond]="${hits[0]}"
  fi
done

if (( missing )); then
  echo "Some checkpoints unresolved — fix the CKPT paths above and rerun." >&2
  exit 1
fi

echo
echo "Extracting (gpu=$GPU, episodes=$EPISODES ${EXTRA[*]:-}) ..."
fail=0
for cond in "${ORDER[@]}"; do
  echo "=== $cond ==="
  if ! ./run_gpu.sh "$GPU" evaluation.card_game.extract_diag_trajectories \
        --checkpoint "${RESOLVED[$cond]}" --condition "$cond" \
        --num-episodes "$EPISODES" "${EXTRA[@]}"; then
    echo "  ✘ $cond FAILED (continuing)"
    fail=1
  fi
done

echo
echo "Done. npz files in evaluation/card_game/diag_data/"
echo "Pull them to your Mac, then score with:"
echo "  uv run --no-project --with dec-pomdp-diagnostics --with numpy \\"
echo "    python evaluation/card_game/run_dec_pomdp_diagnostics.py \\"
echo "    --data-dir evaluation/card_game/diag_data --out evaluation/card_game/diag_data/diagnostics_table.csv"
exit $fail
