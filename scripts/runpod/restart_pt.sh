#!/usr/bin/env bash
# Bootstrap deps + patch Murzik HF checkpoint + launch PT (survives SSH disconnect).
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
export PYTHONPATH="/workspace/NULLXES_MURZIK:${PYTHONPATH:-}"
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ROOT=/workspace/NULLXES_MURZIK
MODEL_DIR="${MODEL_DIR:-/workspace/models/murzik-15b}"
CONFIG="${1:-configs/training/pt_murzik_15b_multilingual.yaml}"

cd "${ROOT}"
git pull --ff-only origin main

pip install -U pip -q
pip install -r runpod/requirements.txt -q
pip install git+https://github.com/hiyouga/LlamaFactory.git -q

cp -f data/dataset_info_multilingual.json /workspace/data/dataset_info.json

rm -rf "${HF_HOME}/modules/transformers_modules/"*murzik* 2>/dev/null || true

python scripts/export_hf_remote_code.py --model-dir "${MODEL_DIR}"

if [[ ! -f "${MODEL_DIR}/murzik.model" ]]; then
  echo "=== Train Murzik SPM tokenizer (wiki stream, no C4 split) ==="
  python scripts/train_murzik_spm.py --model-dir "${MODEL_DIR}" --num-samples 20000
else
  echo "=== Reuse existing Murzik SPM: ${MODEL_DIR}/murzik.model ==="
  python scripts/export_hf_remote_code.py --model-dir "${MODEL_DIR}"
  python - <<'PY'
import sys
from pathlib import Path

model_dir = Path("/workspace/models/murzik-15b")
sys.path.insert(0, "/workspace/NULLXES_MURZIK")
from murzik.tokenization_murzik import MurzikTokenizer

tok = MurzikTokenizer(vocab_file=str(model_dir / "murzik.model"))
tok.save_pretrained(model_dir)
print("[restart_pt] MurzikTokenizer saved")
PY
fi

echo "=== Train: ${CONFIG} ==="
python llamafactory_ext/register_murzik.py
llamafactory-cli train "${CONFIG}"
