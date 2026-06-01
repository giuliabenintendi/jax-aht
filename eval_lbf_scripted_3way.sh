#!/usr/bin/env bash
# Run the scripted-partner eval on three trained LBF 12x12-8food egos:
#   be9uqslh   aux + channel        (aux=0.05, JA_FRUIT_PARTNER_FEED=True,  12 seeds)
#   7hh2s31k   baseline + channel   (aux=0,    JA_FRUIT_PARTNER_FEED=True,  12 seeds)
#   lzbfimyk   baseline + no-channel(aux=0,    JA_FRUIT_PARTNER_FEED=False, 5 seeds)
#
# CSVs land under artifacts/eval_scripted_<run-id>_<partner>_<feed-mode>.csv.
#
# Usage:
#   ./eval_lbf_scripted_3way.sh [device] [partner] [num_episodes]
#     device       : GPU index                              (default 0)
#     partner      : seq_nearest | seq_lex | random | ...   (default seq_nearest)
#     num_episodes : episodes per seed                      (default 256)
#
# Examples:
#   ./eval_lbf_scripted_3way.sh 4 seq_nearest 256
#   ./eval_lbf_scripted_3way.sh 0 random 128

set -e
cd "$(dirname "$0")" || exit 1

device="${1:-0}"
partner="${2:-seq_nearest}"
num_eps="${3:-256}"

# Edit this list if you want to add more runs (e.g., a future aux_nochannel run).
runs=(
    be9uqslh
    7hh2s31k
    lzbfimyk
)

echo "================================================================"
echo "scripted-partner eval"
echo "  device=${device}  partner=${partner}  num_episodes=${num_eps}"
echo "  runs=(${runs[*]})"
echo "================================================================"

for run in "${runs[@]}"; do
    echo
    echo "----------------  run=${run}  ----------------"
    ./run_gpu.sh "${device}" evaluation.eval_scripted_partners \
        --from-wandb --run-id "${run}" \
        --partner "${partner}" \
        --num-episodes "${num_eps}"
done

echo
echo "================================================================"
echo "Done. Per-episode CSVs:"
for run in "${runs[@]}"; do
    echo "  artifacts/eval_scripted_${run}_${partner}_<feed-mode>.csv"
done
echo "================================================================"
