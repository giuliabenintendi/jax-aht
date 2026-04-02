#!/usr/bin/env bash
# Attention-gaze ablations for LBF image tasks.
# Runs 4 conditions sequentially on one GPU:
#   1. baseline          beta=0.0    feed_other_attn=false
#   2. beta only         beta=0.001  feed_other_attn=false
#   3. feed only         beta=0.0    feed_other_attn=true
#   4. beta + feed       beta=0.001  feed_other_attn=true
#
# Usage:
#   ./run_gaze_ablation_overnight.sh <gpu> [task] [timesteps]
#
# Defaults:
#   task=lbf-image-10food
#   timesteps=3e6

set -euo pipefail

GPU="${1:?Usage: ./run_gaze_ablation_overnight.sh <gpu> [task] [timesteps]}"
TASK="${2:-lbf-image-10food}"
TIMESTEPS="${3:-3e6}"
SEEDS=5
LOGDIR="logs"

mkdir -p "${LOGDIR}"

# Gaze currently reuses the lbf-image algorithm profile for both LBF image tasks.
ALG_PROFILE="gaze_ippo/lbf-image"

BASE_CMD="./run_gpu.sh ${GPU} marl.run --config-name base_config_gaze_ippo \
task=${TASK} \
algorithm=${ALG_PROFILE} \
algorithm.TOTAL_TIMESTEPS=${TIMESTEPS} \
algorithm.NUM_SEEDS=${SEEDS} \
logger.mode=online"

LOGFILE="${LOGDIR}/gaze_ablation_${TASK}_$(date +%Y%m%d_%H%M%S).log"

nohup bash -c "
echo \"[\$(date +%H:%M)] task=${TASK} steps=${TIMESTEPS} seeds=${SEEDS}\"

echo \"[\$(date +%H:%M)] 1/4 baseline beta=0.0 feed=false\"
${BASE_CMD} algorithm.GAZE_BETA_MAX=0.0 algorithm.FEED_OTHER_ATTN=false label=gaze_${TASK}_baseline_5s

echo \"[\$(date +%H:%M)] 2/4 beta-only beta=0.001 feed=false\"
${BASE_CMD} algorithm.GAZE_BETA_MAX=0.001 algorithm.FEED_OTHER_ATTN=false label=gaze_${TASK}_beta0.001_5s

echo \"[\$(date +%H:%M)] 3/4 feed-only beta=0.0 feed=true\"
${BASE_CMD} algorithm.GAZE_BETA_MAX=0.0 algorithm.FEED_OTHER_ATTN=true label=gaze_${TASK}_feed_5s

echo \"[\$(date +%H:%M)] 4/4 beta+feed beta=0.001 feed=true\"
${BASE_CMD} algorithm.GAZE_BETA_MAX=0.001 algorithm.FEED_OTHER_ATTN=true label=gaze_${TASK}_beta0.001_feed_5s

echo \"[\$(date +%H:%M)] Done\"
" > "${LOGFILE}" 2>&1 &

echo "Launched gaze ablations on GPU ${GPU}"
echo "Log: ${LOGFILE}"
