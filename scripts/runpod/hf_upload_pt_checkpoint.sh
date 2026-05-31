#!/usr/bin/env bash
# Upload Murzik-15B PT checkpoint to Hugging Face.
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN first}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
HF_REPO="${HF_REPO:-MagistrTheOne/murzik-15b-pt-pilot}"
CKPT_DIR="${CKPT_DIR:-/workspace/checkpoints/pt-15b-multilingual-2x/checkpoint-1500}"

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "Checkpoint not found: $CKPT_DIR" >&2
  exit 1
fi

hf auth login --token "$HF_TOKEN"
hf repo create "$HF_REPO" --type model --no-private 2>/dev/null || true
hf upload "$HF_REPO" "$CKPT_DIR" . --repo-type model --no-private \
  --commit-message "Murzik-15B multilingual PT pilot (1500 steps, wiki+identity)"
echo "https://huggingface.co/${HF_REPO}"
