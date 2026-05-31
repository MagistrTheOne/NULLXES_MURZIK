#!/usr/bin/env bash
# Upload Murzik-15B PT checkpoint to Hugging Face (default: murzik-15b-init).
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN first}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HF_REPO="${HF_REPO:-MagistrTheOne/murzik-15b-init}"
CKPT_DIR="${CKPT_DIR:-/workspace/checkpoints/pt-15b-multilingual-2x}"
README_SRC="${README_SRC:-${REPO_ROOT}/hf_model_card/README_pt_on_init.md}"

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "Checkpoint not found: $CKPT_DIR" >&2
  exit 1
fi
if [[ ! -f "$README_SRC" ]]; then
  echo "Model card not found: $README_SRC" >&2
  exit 1
fi

# LlamaFactory may write README with invalid base_model (local path). Overwrite before upload.
cp "$README_SRC" "$CKPT_DIR/README.md"

hf auth login --token "$HF_TOKEN"
hf repo create "$HF_REPO" --type model --no-private 2>/dev/null || true
hf upload "$HF_REPO" "$CKPT_DIR" . --repo-type model --no-private \
  --commit-message "Murzik-15B multilingual PT pilot (1500 steps, wiki+identity)" \
  --exclude "optimizer.pt" \
  --exclude "scheduler.pt" \
  --exclude "rng_state*.pth" \
  --exclude "trainer_state.json" \
  --exclude "training_args.bin" \
  --exclude "checkpoint-*"
echo "https://huggingface.co/${HF_REPO}"
