#!/usr/bin/env bash
# Launch LlamaFactory PT on RunPod Instant Cluster (run on ALL nodes).
set -euo pipefail

CONFIG="${1:-configs/training/pt_murzik_32b.yaml}"

source /workspace/NULLXES_MURZIK/scripts/runpod/bootstrap_node.sh 2>/dev/null || true

MASTER_ADDR="${PRIMARY_ADDR:-${MASTER_ADDR}}"
MASTER_PORT="${PRIMARY_PORT:-${MASTER_PORT:-29500}}"
NNODES="${NUM_NODES:-1}"
NPROC="${NUM_TRAINERS:-8}"
NODE_RANK="${NODE_RANK:-0}"

cd /workspace/NULLXES_MURZIK

echo "Launching PT with config=${CONFIG}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  llamafactory-cli train "${CONFIG}"
