#!/usr/bin/env bash
# Usage: ./run_gpu.sh [device] module [args...]

DEFAULTVALUE=0
device="${1:-$DEFAULTVALUE}"
shift 2>/dev/null

CUDA_VISIBLE_DEVICES="${device}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false" \
LD_LIBRARY_PATH="" \
nice -n 5 \
uv run python -m "$@"
