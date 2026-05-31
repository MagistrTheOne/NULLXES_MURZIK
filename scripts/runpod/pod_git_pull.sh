#!/usr/bin/env bash
# Safe git pull on RunPod when local untracked files block merge.
set -euo pipefail
ROOT="${1:-/workspace/NULLXES_MURZIK}"
cd "$ROOT"
if git diff --quiet && git diff --cached --quiet; then
  :
else
  echo "Stashing local modifications..."
  git stash push -u -m "pod-auto-stash $(date -u +%Y%m%dT%H%M%SZ)" || true
fi
# Drop untracked files that also exist upstream (common on pods).
for f in \
  configs/training/sft_murzik_15b_identity_smoke.yaml \
  configs/training/sft_murzik_15b_identity_hard.yaml
do
  if [[ -f "$f" ]] && git ls-tree -r origin/main --name-only | grep -qx "$f"; then
    rm -f "$f"
  fi
done
git pull --ff-only
echo "At $(git rev-parse --short HEAD): $(git log -1 --pretty=%s)"
