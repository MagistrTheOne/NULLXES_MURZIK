#!/usr/bin/env bash
# NULLXES foundation PT bootstrap (no HF wiki/c4). Run on RunPod only.
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
export PYTHONPATH="/workspace/NULLXES_MURZIK:${PYTHONPATH:-}"
export FORCE_TORCHRUN=1

ROOT="/workspace/NULLXES_MURZIK"
DATA="/workspace/data"
MODEL="/workspace/models/murzik-15b"
TOK="${DATA}/tokenizer/murzik-spm128k.model"
CONFIG="${1:-configs/training/pt_murzik_15b_foundation.yaml}"
LOG="${LOG:-/workspace/logs/pt-15b-foundation.log}"

mkdir -p "${DATA}/pt/shards" "${HF_HOME}" "$(dirname "${LOG}")"
cd "${ROOT}"
git pull --ff-only origin main || true

pip install -U pip -q
pip install -r runpod/requirements.txt -q
pip install git+https://github.com/hiyouga/LlamaFactory.git -q

echo "=== Build NULLXES foundation corpus ==="
python scripts/build_foundation_corpus.py \
  --manifest data/foundation_manifest.json \
  --out-dir "${DATA}" \
  --copy-examples

python scripts/validate_pt_readiness.py --data-dir "${DATA}" --min-docs 50 || true

cp -f data/dataset_info_foundation.json "${DATA}/dataset_info.json"

if [[ ! -f "${TOK}" ]]; then
  python scripts/train_tokenizer.py \
    --corpus "${DATA}/pt/murzik_pt.jsonl" \
    --out "${TOK}"
fi

if [[ ! -f "${MODEL}/model.safetensors" && ! -f "${MODEL}/model.safetensors.index.json" ]]; then
  echo "=== Fresh init (initializer_range 0.006) ==="
  python scripts/init_model.py \
    --config configs/model/murzik_15b_pilot.json \
    --out "${MODEL}" \
    --tokenizer "${TOK}"
else
  echo "=== Model exists at ${MODEL} — continuing weights ==="
  python scripts/patch_model_repo.py --model-dir "${MODEL}" --tokenizer "${TOK}"
fi

echo "=== Train: ${CONFIG} ===" | tee -a "${LOG}"
nohup bash scripts/llamafactory_train.sh "${CONFIG}" >> "${LOG}" 2>&1 &
echo "PID $! — tail -f ${LOG}"
echo "QA after checkpoint: python scripts/runpod/qa_checkpoint_ladder.py"
