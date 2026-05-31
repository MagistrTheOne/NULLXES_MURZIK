#!/usr/bin/env bash
# Wait for PT torchrun to finish, then run smoke QA and write summary logs.
set -euo pipefail

LOG_PT=/workspace/logs/pt-15b-2x.log
LOG_QA=/workspace/logs/qa-pt-smoke.log
LOG_SUM=/workspace/logs/pt-run-summary.txt
CKPT=/workspace/checkpoints/pt-15b-multilingual-2x/checkpoint-1500
REPO=/workspace/NULLXES_MURZIK

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HF_HOME=/workspace/cache/huggingface

echo "=== wait_pt_and_qa started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG_SUM"

if pgrep -f "torchrun.*pt_murzik_15b_fast_2x" >/dev/null 2>&1; then
  echo "Waiting for torchrun to exit..." | tee -a "$LOG_SUM"
  while pgrep -f "torchrun.*pt_murzik_15b_fast_2x" >/dev/null 2>&1; do
    step=$(grep -oE '[0-9]+/1500' "$LOG_PT" 2>/dev/null | tail -1 || true)
    echo "$(date -u +%H:%M:%S) still training ${step:-?}" | tee -a "$LOG_SUM"
    sleep 60
  done
  echo "Training process finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_SUM"
else
  echo "Training already finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_SUM"
fi

echo "--- last 30 log lines ---" | tee -a "$LOG_SUM"
tail -30 "$LOG_PT" | tee -a "$LOG_SUM"

if [[ ! -d "$CKPT" ]]; then
  echo "ERROR: checkpoint dir missing: $CKPT" | tee -a "$LOG_SUM"
  exit 1
fi

echo "Checkpoint files:" | tee -a "$LOG_SUM"
ls -lah "$CKPT" | tee -a "$LOG_SUM"

echo "Running QA smoke..." | tee -a "$LOG_SUM"
cd "$REPO"
python scripts/runpod/qa_pt_smoke.py --checkpoint "$CKPT" 2>&1 | tee "$LOG_QA"
echo "QA log: $LOG_QA" | tee -a "$LOG_SUM"

echo "=== wait_pt_and_qa done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG_SUM"
