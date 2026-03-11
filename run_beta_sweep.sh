#!/usr/bin/env bash
# Beta sweep for JA-IPPO across Overcooked layouts.
# Runs 5 beta values × 3 layouts = 15 runs across 3 GPUs.
#
# Usage: ./run_beta_sweep.sh
# Or to run a single GPU's jobs: ./run_beta_sweep.sh <gpu_index>
#   gpu_index: 0, 1, or 2 (maps to CUDA devices 1, 5, 6)

set -e

GPUS=(1 5)
TIMESTEPS=5e6
WARMUP=3500000
BETAS=(1.0 2.5)
LAYOUTS=(cramped_room forced_coord coord_ring)

# Build list of (beta, layout) jobs
JOBS=()
for BETA in "${BETAS[@]}"; do
    for LAYOUT in "${LAYOUTS[@]}"; do
        JOBS+=("${BETA}:${LAYOUT}")
    done
done

# Distribute jobs round-robin across GPUs
run_gpu_jobs() {
    local gpu_idx=$1
    local gpu_device=${GPUS[$gpu_idx]}
    local total_gpus=${#GPUS[@]}
    local job_count=0

    for i in "${!JOBS[@]}"; do
        if (( i % total_gpus == gpu_idx )); then
            IFS=':' read -r BETA LAYOUT <<< "${JOBS[$i]}"
            job_count=$((job_count + 1))
            echo "[GPU ${gpu_device}] Job ${job_count}: ${LAYOUT} BETA=${BETA}"
            ./run_gpu.sh "$gpu_device" marl.run \
                -cn base_config_ja_ippo \
                task=overcooked-v1/$LAYOUT \
                algorithm.TOTAL_TIMESTEPS=$TIMESTEPS \
                algorithm.JA_WARMUP_ENV_STEPS=$WARMUP \
                algorithm.JA_BETA_MAX=$BETA \
                label=beta_sweep
        fi
    done
    echo "[GPU ${gpu_device}] All jobs complete."
}

if [ -n "$1" ]; then
    # Run a single GPU's jobs
    run_gpu_jobs "$1"
else
    # Launch all GPUs in parallel
    for gpu_idx in $(seq 0 $((${#GPUS[@]} - 1))); do
        run_gpu_jobs "$gpu_idx" &
    done
    wait
    echo "All sweep runs complete."
fi
