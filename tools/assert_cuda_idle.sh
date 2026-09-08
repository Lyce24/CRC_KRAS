#!/usr/bin/env bash
set -euo pipefail

nvidia_smi="/usr/lib/wsl/lib/nvidia-smi"
if [[ ! -x "${nvidia_smi}" ]]; then
    echo "CUDA preflight failed: ${nvidia_smi} is unavailable" >&2
    exit 1
fi

compute_pids="$(${nvidia_smi} --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | /usr/bin/awk 'NF {gsub(/[[:space:]]/, "", $0); if ($0 != "") print $0}')"
if [[ -n "${compute_pids}" ]]; then
    echo "CUDA preflight failed: compute processes still own the GPU:" >&2
    echo "${compute_pids}" >&2
    exit 1
fi

echo "CUDA preflight passed: no compute process owns the GPU"
