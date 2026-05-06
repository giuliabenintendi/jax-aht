#!/usr/bin/env bash
# Post-hoc analysis for the May 2026 card-game JA-card-JSD sweep:
#   - per-seed per-agent attention plots (analyze_attention.py --all-seeds)
#       -> /scratch/benintendi/jax-aht/plots/card_game/<label>_<timestamp>/
#   - per-seed per-agent attention videos for the best run only
#     (add_eval_videos.py, also re-logs them to the wandb run)
#       -> <run_root>/videos/seed_<i>/eval_attention_{agent0,agent1,combined}.mp4
#
# Usage:
#   ./run_card_game_ja_postprocess.sh <gpu>           # plots-only (all 6 runs) + videos for best run
#   PLOTS_ONLY=1 ./run_card_game_ja_postprocess.sh <gpu>   # skip videos
#   VIDEOS_ONLY=1 ./run_card_game_ja_postprocess.sh <gpu>  # skip plots
#
# Each run takes a few minutes; expect ~20-30 min total on a single GPU.

GPU="${1:?Usage: ./run_card_game_ja_postprocess.sh <gpu>}"
PLOTS_ONLY="${PLOTS_ONLY:-0}"
VIDEOS_ONLY="${VIDEOS_ONLY:-0}"

PLOT_ROOT="/scratch/benintendi/jax-aht/plots/card_game"
RES_ROOT="/scratch/benintendi/jax-aht/results/card-game/ja_ippo"

# (run_root_subpath, wandb_id, short_desc) for the 6 JA-card-JSD runs.
# run_root = $RES_ROOT/<subpath>; checkpoint = $run_root/saved_train_run.
RUNS=(
  "ja_card_jsd_only_0.02/2026-05-05_16-06-20  jdyyp78w  jsd0.02_3s_5M"
  "ja_card_jsd_only_0.05/2026-05-05_17-02-37  tf2d8mo6  jsd0.05_3s_5M"
  "ja_card_jsd_only_0.1/2026-05-05_17-59-08   ps2w3xd4  jsd0.1_3s_5M"
  "ja_card_jsd_only_0.2/2026-05-05_18-55-21   o4m2r97l  jsd0.2_3s_5M"
  "ja_card_jsd_only_0.5/2026-05-05_19-51-36   9bva9xxt  jsd0.5_3s_5M"
  "ja_card_jsd0.1_12s_8M/2026-05-05_20-40-36  z32brv0i  jsd0.1_12s_8M_BEST"
)
BEST_INDEX=5  # zero-based index into RUNS for the best run (sith-admiral-1264)

mkdir -p logs
LOG="logs/card_game_ja_postprocess_$(date +%Y%m%d_%H%M%S).log"

run_plots () {
  local subpath="$1" rid="$2" desc="$3"
  local run_root="${RES_ROOT}/${subpath}"
  local ckpt="${run_root}/saved_train_run"
  local label_ts="$(basename "$(dirname "${subpath}")")_$(basename "${subpath}")"
  local out_dir="${PLOT_ROOT}/${label_ts}"

  if [[ ! -d "${ckpt}" ]]; then
    echo "[$(date +%H:%M)] SKIP plots ${desc} (${rid}): no checkpoint at ${ckpt}"
    return
  fi
  echo "[$(date +%H:%M)] PLOTS ${desc} (${rid}) -> ${out_dir}"
  mkdir -p "${out_dir}"
  ./run_gpu.sh "${GPU}" evaluation.analyze_attention \
      --checkpoint "${ckpt}" \
      --all-seeds \
      --num-episodes 5 \
      --output-dir "${out_dir}"
}

run_videos () {
  local subpath="$1" rid="$2" desc="$3"
  local run_root="${RES_ROOT}/${subpath}"
  local ckpt="${run_root}/saved_train_run"

  if [[ ! -d "${ckpt}" ]]; then
    echo "[$(date +%H:%M)] SKIP videos ${desc} (${rid}): no checkpoint at ${ckpt}"
    return
  fi
  echo "[$(date +%H:%M)] VIDEOS ${desc} (${rid}) -> ${run_root}/videos/"
  ./run_gpu.sh "${GPU}" evaluation.add_eval_videos \
      --checkpoint "${ckpt}" \
      --run-id "${rid}"
}

{
  echo "[$(date +%H:%M)] post-process start, GPU=${GPU}, log=${LOG}"

  if [[ "${VIDEOS_ONLY}" != "1" ]]; then
    for entry in "${RUNS[@]}"; do
      read -r subpath rid desc <<< "${entry}"
      run_plots "${subpath}" "${rid}" "${desc}"
    done
  fi

  if [[ "${PLOTS_ONLY}" != "1" ]]; then
    read -r subpath rid desc <<< "${RUNS[$BEST_INDEX]}"
    run_videos "${subpath}" "${rid}" "${desc}"
  fi

  echo "[$(date +%H:%M)] post-process done"
} 2>&1 | tee "${LOG}"
