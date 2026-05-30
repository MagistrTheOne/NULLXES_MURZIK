#!/usr/bin/env bash
# Production bootstrap on RunPod (no demo/smoke).
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/NULLXES_MURZIK:${PYTHONPATH:-}
export HF_HOME=/workspace/cache/huggingface

ROOT=/workspace/NULLXES_MURZIK
DATA=/workspace/data
MODELS=/workspace/models

mkdir -p "${HF_HOME}" "${DATA}/pt" "${DATA}/sft" "${DATA}/dpo" "${MODELS}"

cd "${ROOT}"
git pull --ff-only origin main

pip install -q -U pip
pip install -q -r runpod/requirements.txt
pip install -q git+https://github.com/hiyouga/LlamaFactory.git

cp -f data/dataset_info.json "${DATA}/dataset_info.json"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: set HF_TOKEN before running bootstrap"
  exit 1
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"
if command -v hf >/dev/null 2>&1; then
  hf auth login --token "${HUGGING_FACE_HUB_TOKEN}" --add-to-git-credential
else
  huggingface-cli login --token "${HUGGING_FACE_HUB_TOKEN}" --add-to-git-credential
fi

echo "=== Init MURZIK-15B weights ==="
python scripts/init_model.py \
  --config configs/model/murzik_15b_pilot.json \
  --out "${MODELS}/murzik-15b"

python llamafactory_ext/register_murzik.py

HF_REPO="${HF_REPO:-MagistrTheOne/murzik-15b-init}"
echo "=== Upload to Hugging Face: ${HF_REPO} ==="
if command -v hf >/dev/null 2>&1; then
  hf upload "${HF_REPO}" "${MODELS}/murzik-15b" . --repo-type model --private --create-repo
else
  huggingface-cli upload "${HF_REPO}" "${MODELS}/murzik-15b" . --repo-type model --private --create-repo
fi

echo "=== Done ==="
echo "Model local:  ${MODELS}/murzik-15b"
echo "Model HF:     https://huggingface.co/${HF_REPO}"
echo "Next: put PT data in ${DATA}/pt/murzik_pt.jsonl then:"
echo "  bash scripts/llamafactory_train.sh configs/training/pt_murzik_15b.yaml"
