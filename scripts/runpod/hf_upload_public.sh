#!/usr/bin/env bash
# One-shot public upload of MURZIK-15B to Hugging Face.
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN first}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
HF_REPO="${HF_REPO:-MagistrTheOne/murzik-15b-init}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/murzik-15b}"

hf auth login --token "$HF_TOKEN"
hf repo create "$HF_REPO" --type model --no-private 2>/dev/null || true
hf upload "$HF_REPO" "$MODEL_DIR" . --repo-type model --no-private --commit-message "MURZIK-15B init weights"
echo "https://huggingface.co/${HF_REPO}"
