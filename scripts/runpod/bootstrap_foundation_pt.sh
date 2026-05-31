#!/usr/bin/env bash
# NULLXES foundation PT on 2× H200 — corpus + train in one shot.
set -euo pipefail

export PYTHONPATH="/workspace/NULLXES_MURZIK:${PYTHONPATH:-}"
export FORCE_TORCHRUN=1

ROOT="/workspace/NULLXES_MURZIK"
DATA="/workspace/data"
MODEL="/workspace/models/murzik-15b"
TOK="${DATA}/tokenizer/murzik-spm128k.model"
MANIFEST="${MANIFEST:-data/foundation_manifest_language_core.json}"
CONFIG="${1:-configs/training/pt_murzik_15b_2x.yaml}"
LOG="${LOG:-/workspace/logs/pt-15b-2x.log}"

mkdir -p "${DATA}" "$(dirname "${LOG}")"
cd "${ROOT}"
git pull --ff-only origin main || true

pip install -U pip -q
pip install -r runpod/requirements.txt -q
pip install git+https://github.com/hiyouga/LlamaFactory.git -q

echo "=== Seed / refresh shard base (optional) ==="
python scripts/seed_corpus_base.py --scale 2 || true

echo "=== Build merged PT corpus (${MANIFEST}) ==="
python scripts/build_foundation_corpus.py \
  --manifest "${MANIFEST}" \
  --out-dir "${DATA}" \
  --copy-shards

python scripts/validate_pt_readiness.py --data-dir "${DATA}" --min-docs 1000 || true

if [[ ! -f "${TOK}" ]]; then
  python scripts/train_tokenizer.py --corpus "${DATA}/pt/murzik_pt.jsonl" --out "${TOK}"
fi

if [[ ! -f "${MODEL}/model.safetensors" && ! -f "${MODEL}/model.safetensors.index.json" ]]; then
  echo "=== Fresh init (initializer_range 0.006) ==="
  python scripts/init_model.py \
    --config configs/model/murzik_15b_pilot.json \
    --out "${MODEL}" \
    --tokenizer "${TOK}"
else
  python scripts/patch_model_repo.py --model-dir "${MODEL}" --tokenizer "${TOK}"
fi

echo "=== Train 2× GPU: ${CONFIG} ===" | tee -a "${LOG}"
nohup bash scripts/llamafactory_train.sh "${CONFIG}" >> "${LOG}" 2>&1 &
echo "PID $! — tail -f ${LOG}"
echo "QA: python scripts/runpod/qa_checkpoint_ladder.py --stages pt=/workspace/checkpoints/pt-15b-2x"
