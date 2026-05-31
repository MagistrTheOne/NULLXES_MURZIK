#!/usr/bin/env bash
# 2× H200 bootstrap: HF transfer, identity corpus, continue PT from pilot checkpoint.
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
export PYTHONPATH="/workspace/NULLXES_MURZIK:${PYTHONPATH:-}"
export FORCE_TORCHRUN=1
export HF_HUB_ENABLE_HF_TRANSFER=1

ROOT="/workspace/NULLXES_MURZIK"
DATA="/workspace/data"
MODEL="/workspace/models/murzik-15b"
PILOT="/workspace/checkpoints/pt-15b-multilingual"
TOK="${DATA}/tokenizer/murzik-spm128k.model"
CONFIG="${1:-configs/training/pt_murzik_15b_multilingual_2x.yaml}"
LOG="${LOG:-/workspace/logs/pt-15b-2x.log}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: export HF_TOKEN=... before bootstrap (do not commit tokens)"
  exit 1
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"

mkdir -p "${DATA}/pt" "${HF_HOME}" "$(dirname "${LOG}")"
cd "${ROOT}"
git pull --ff-only origin main || true

pip install -U pip -q
pip install -r runpod/requirements.txt -q
pip install -U "huggingface_hub[hf_transfer]" bitsandbytes -q
pip install git+https://github.com/hiyouga/LlamaFactory.git -q

if command -v hf >/dev/null 2>&1; then
  hf auth login --token "${HUGGING_FACE_HUB_TOKEN}"
else
  huggingface-cli login --token "${HUGGING_FACE_HUB_TOKEN}"
fi

cp -f data/dataset_info_multilingual.json "${DATA}/dataset_info.json"
cp -f data/examples/murzik_identity.jsonl "${DATA}/pt/murzik_identity.jsonl"

# Repeat identity corpus 20× so PT sees NULLXES branding often
python - <<'PY'
from pathlib import Path
src = Path("/workspace/data/pt/murzik_identity.jsonl")
lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
Path("/workspace/data/pt/murzik_identity.jsonl").write_text(
    "\n".join(lines * 20) + "\n", encoding="utf-8"
)
print(f"murzik_identity: {len(lines)} unique -> {len(lines)*20} rows")
PY

if [[ ! -f "${TOK}" ]]; then
  python scripts/train_tokenizer.py --out "${TOK}"
fi

BASE="${PILOT}"
if [[ ! -f "${PILOT}/model.safetensors" && ! -f "${PILOT}/model.safetensors.index.json" ]]; then
  BASE="${MODEL}"
  if [[ ! -f "${MODEL}/model.safetensors" ]]; then
    hf download MagistrTheOne/murzik-15b-init --local-dir "${MODEL}"
  fi
  python scripts/patch_model_repo.py --model-dir "${MODEL}" --tokenizer "${TOK}"
else
  echo "=== Continue from pilot checkpoint ${PILOT} ==="
  cp -f "${TOK}" "${PILOT}/murzik.model" 2>/dev/null || true
  python scripts/patch_model_repo.py --model-dir "${PILOT}" --tokenizer "${TOK}" || true
fi

sed -i "s|^model_name_or_path:.*|model_name_or_path: ${BASE}|" "${CONFIG}" || true

echo "=== Train 2× GPU: ${CONFIG} ===" | tee -a "${LOG}"
nohup bash scripts/llamafactory_train.sh "${CONFIG}" >> "${LOG}" 2>&1 &
echo "PID $! — tail -f ${LOG}"
