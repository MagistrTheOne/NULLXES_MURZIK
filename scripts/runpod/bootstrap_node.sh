#!/usr/bin/env bash
# Run on every Instant Cluster node before training.
set -euo pipefail

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

echo "=== MURZIK bootstrap ==="
echo "NODE_RANK=${NODE_RANK:-?} NUM_NODES=${NUM_NODES:-?} WORLD_SIZE=${WORLD_SIZE:-?}"
echo "PRIMARY_ADDR=${PRIMARY_ADDR:-${MASTER_ADDR:-?}}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"

cd /workspace/NULLXES_MURZIK || cd /workspace

# Optional: mount HF cache on volume
export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
mkdir -p "${HF_HOME}"

echo "Bootstrap complete."
