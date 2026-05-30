#!/usr/bin/env bash
# Install multilingual dataset registry on RunPod Volume.
set -euo pipefail

DATA="${1:-/workspace/data}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "${DATA}/pt" "${DATA}/sft" "${DATA}/dpo"

# Merge multilingual HF dataset definitions (keep local murzik_* entries)
cp -f "${ROOT}/data/dataset_info_multilingual.json" "${DATA}/dataset_info.json"

echo "Installed ${DATA}/dataset_info.json"
echo ""
echo "Multilingual PT datasets (HF hub, no manual download):"
echo "  wiki_en, wiki_ru, wiki_de, wiki_es, wiki_fr, wiki_zh, wiki_uk"
echo "  mc4_en, mc4_ru"
echo ""
echo "Multilingual SFT:"
echo "  aya_sft  -> CohereLabs/aya_dataset (204k, 65 languages)"
echo ""
echo "Train PT:"
echo "  bash scripts/llamafactory_train.sh configs/training/pt_murzik_15b_multilingual.yaml"
echo ""
echo "Train SFT:"
echo "  bash scripts/llamafactory_train.sh configs/training/sft_murzik_15b_multilingual.yaml"
