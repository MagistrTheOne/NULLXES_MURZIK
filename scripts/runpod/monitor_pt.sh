#!/usr/bin/env bash
# Start PT and poll log until first loss or failure (RunPod).
set -euo pipefail

LOG="${LOG:-/workspace/logs/pt-15b-multilingual.log}"
ROOT="${ROOT:-/workspace/NULLXES_MURZIK}"
CONFIG="${CONFIG:-configs/training/pt_murzik_15b_multilingual.yaml}"
MAX_WAIT="${MAX_WAIT:-900}"

cd "${ROOT}"
git pull --ff-only || true

echo "=== $(date -Is) install bitsandbytes if missing ==="
python -c "import bitsandbytes" 2>/dev/null || pip install -q -U bitsandbytes

pkill -f 'torchrun.*pt_murzik_15b_multilingual' 2>/dev/null || true
sleep 3

echo "=== $(date -Is) PT monitor start ===" >> "${LOG}"
nohup bash scripts/llamafactory_train.sh "${CONFIG}" >> "${LOG}" 2>&1 &
echo "Started PID $!"

deadline=$((SECONDS + MAX_WAIT))
while (( SECONDS < deadline )); do
  sleep 30
  if ! pgrep -f 'torchrun.*pt_murzik_15b_multilingual' >/dev/null 2>&1; then
    if grep -q "'loss':" "${LOG}" 2>/dev/null || grep -q '"loss":' "${LOG}" 2>/dev/null; then
      echo "OK: training finished or logged loss"
      tail -5 "${LOG}"
      exit 0
    fi
    echo "FAIL: torchrun exited without loss"
    tail -40 "${LOG}"
    exit 1
  fi
  if grep -qE "'loss':|\"loss\":" "${LOG}" 2>/dev/null; then
    echo "OK: first loss logged"
    grep -E "loss|Running training|Pure bf16" "${LOG}" | tail -8
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
    exit 0
  fi
  echo "$(date -Is) waiting... $(pgrep -af torchrun | head -1 || echo no process)"
done

echo "TIMEOUT after ${MAX_WAIT}s"
tail -20 "${LOG}"
exit 2
