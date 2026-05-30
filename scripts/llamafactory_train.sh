#!/usr/bin/env bash
# Wrapper: register Murzik template, then run LlamaFactory train.
set -euo pipefail

CONFIG="${1:?Usage: llamafactory_train.sh configs/training/xxx.yaml}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
export NULLXES_PROJECT="${NULLXES_PROJECT:-MURZIK}"
export NULLXES_CONTACT="${NULLXES_CONTACT:-ceo@nullxes.com}"

cd "${ROOT}"

echo "=== Register Murzik in LlamaFactory ==="
python llamafactory_ext/register_murzik.py

echo "=== Train: ${CONFIG} ==="
llamafactory-cli train "${CONFIG}"
