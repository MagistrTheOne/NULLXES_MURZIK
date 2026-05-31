#!/usr/bin/env python3
"""Patch model dir: Transformers 5 auto_map, murzik code, real SPM tokenizer."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.murzik_hf_export import auto_map_for, export_murzik_code  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer", required=True, help="Path to .model SentencePiece file")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tok_path = Path(args.tokenizer)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if not tok_path.is_file():
        raise FileNotFoundError(tok_path)

    cfg_path = model_dir / "config.json"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type", "murzik")

    export_murzik_code(model_dir, model_type=model_type)
    cfg["auto_map"] = auto_map_for(model_type)
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    from murzik.tokenization_murzik import MurzikTokenizer

    tokenizer = MurzikTokenizer(vocab_file=str(tok_path))
    tokenizer.save_pretrained(model_dir)

    tok_cfg_path = model_dir / "tokenizer_config.json"
    with tok_cfg_path.open(encoding="utf-8") as f:
        tok_cfg = json.load(f)
    tok_cfg.pop("auto_map", None)
    tok_cfg["tokenizer_class"] = "MurzikTokenizer"
    tok_cfg["vocab_file"] = "murzik.model"
    tok_cfg["use_fast"] = False
    with tok_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2)

    fast_tok = model_dir / "tokenizer.json"
    if fast_tok.is_file():
        fast_tok.unlink()

    sample = tokenizer.encode("Murzik multilingual pre-training.", add_special_tokens=False)
    if len(sample) < 2:
        raise RuntimeError(f"Tokenizer sanity check failed, got ids={sample}")

    print(f"Patched {model_dir}")
    print(f"  auto_map: module.Class (Transformers 5.x)")
    print(f"  tokenizer: murzik.model ({tokenizer.vocab_size} pieces)")


if __name__ == "__main__":
    main()
