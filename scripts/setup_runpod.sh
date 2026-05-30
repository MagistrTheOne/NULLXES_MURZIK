#!/usr/bin/env bash
# One-shot RunPod setup (no Docker). Run after pod start.
set -euo pipefail

ROOT="/workspace/NULLXES_MURZIK"
DATA="/workspace/data"

echo "=== Clone / update repo ==="
if [ ! -d "${ROOT}" ]; then
  git clone https://github.com/MagistrTheOne/NULLXES_MURZIK.git "${ROOT}"
fi
cd "${ROOT}"

echo "=== Install deps ==="
pip install -U pip
pip install -r runpod/requirements.txt
pip install git+https://github.com/hiyouga/LlamaFactory.git
pip install flash-attn --no-build-isolation || echo "flash-attn skipped"

echo "=== Prepare demo dataset ==="
python scripts/prepare_dataset.py --out-dir "${DATA}" --copy-examples
cp data/dataset_info.json "${DATA}/dataset_info.json"

echo "=== Init 15B pilot model ==="
python scripts/init_model.py \
  --config configs/model/murzik_15b_pilot.json \
  --out /workspace/models/murzik-15b

echo "=== Ready. Smoke PT: ==="
echo "  bash scripts/llamafactory_train.sh configs/training/pt_murzik_15b_pilot.yaml"
