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


def auto_map_for(model_type: str) -> dict:
    if model_type == "murzik_moe":
        return {
            "AutoConfig": ["murzik/configuration_murzik_moe.py", "MurzikMoeConfig"],
            "AutoModelForCausalLM": ["murzik/modeling_murzik_moe.py", "MurzikMoeForCausalLM"],
            "AutoTokenizer": ["murzik/tokenization_murzik.py", "MurzikTokenizer"],
        }
    return {
        "AutoConfig": ["murzik/configuration_murzik.py", "MurzikConfig"],
        "AutoModelForCausalLM": ["murzik/modeling_murzik.py", "MurzikForCausalLM"],
        "AutoTokenizer": ["murzik/tokenization_murzik.py", "MurzikTokenizer"],
    }


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

    shutil.copytree(ROOT / "murzik", model_dir / "murzik", dirs_exist_ok=True)

    cfg_path = model_dir / "config.json"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["auto_map"] = auto_map_for(cfg.get("model_type", "murzik"))
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    from murzik.tokenization_murzik import MurzikTokenizer

    tokenizer = MurzikTokenizer(vocab_file=str(tok_path))
    tokenizer.save_pretrained(model_dir)

    # Ensure tokenizer auto_map for trust_remote_code loads.
    tok_cfg_path = model_dir / "tokenizer_config.json"
    with tok_cfg_path.open(encoding="utf-8") as f:
        tok_cfg = json.load(f)
    tok_cfg["auto_map"] = {
        "AutoTokenizer": ["murzik/tokenization_murzik.py", "MurzikTokenizer"]
    }
    with tok_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2)

    print(f"Patched {model_dir}")
    print(f"  auto_map: list format (Transformers 5.x)")
    print(f"  tokenizer: {tok_path.name}")


if __name__ == "__main__":
    main()
