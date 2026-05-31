#!/usr/bin/env bash
# Bootstrap deps + SPM tokenizer + patched model + PT (survives SSH disconnect).
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
export PYTHONPATH="/workspace/NULLXES_MURZIK:${PYTHONPATH:-}"
export FORCE_TORCHRUN=1

ROOT="/workspace/NULLXES_MURZIK"
DATA="/workspace/data"
MODEL="/workspace/models/murzik-15b"
TOK="${DATA}/tokenizer/murzik-spm128k.model"
CONFIG="${1:-configs/training/pt_murzik_15b_multilingual.yaml}"

cd "${ROOT}"
git pull --ff-only origin main || true

pip install -U pip -q
pip install -r runpod/requirements.txt -q
pip install git+https://github.com/hiyouga/LlamaFactory.git -q
pip install flash-attn --no-build-isolation -q || echo "[warn] flash-attn build skipped"

cp -f data/dataset_info_multilingual.json "${DATA}/dataset_info.json"

if [[ ! -f "${TOK}" ]]; then
  echo "=== Train Murzik SPM tokenizer (wiki cache) ==="
  python scripts/train_tokenizer.py --out "${TOK}"
fi

if [[ ! -d "${MODEL}" ]] || [[ ! -f "${MODEL}/model.safetensors.index.json" && ! -f "${MODEL}/model.safetensors" ]]; then
  echo "=== Download HF weights ==="
  hf download MagistrTheOne/murzik-15b-init --local-dir "${MODEL}"
fi

echo "=== Patch model (auto_map + murzik code + tokenizer) ==="
python scripts/patch_model_repo.py --model-dir "${MODEL}" --tokenizer "${TOK}"

echo "=== Train: ${CONFIG} ==="
bash scripts/llamafactory_train.sh "${CONFIG}"
