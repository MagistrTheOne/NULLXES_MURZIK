#!/usr/bin/env bash
# Start NULLXES MURZIK PT in background on RunPod.
set -euo pipefail

CONFIG="${1:-configs/training/nullxes_murzik_15b_pt.yaml}"
LOG="${2:-/workspace/logs/nullxes-pt.log}"
ROOT="/workspace/NULLXES_MURZIK"

mkdir -p /workspace/logs /workspace/checkpoints
cd "${ROOT}"
git pull --ff-only origin main || true
bash scripts/setup_multilingual_datasets.sh /workspace/data

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export NULLXES_PROJECT=MURZIK
export NULLXES_CONTACT=ceo@nullxes.com

nohup bash scripts/llamafactory_train.sh "${CONFIG}" > "${LOG}" 2>&1 &
echo "PID=$!"
echo "LOG=${LOG}"
echo "tail -f ${LOG}"
