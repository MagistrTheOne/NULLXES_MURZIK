#!/usr/bin/env python3
"""
Export Murzik / MurzikMoE HF checkpoint from JSON config (random init).

Usage:
  python scripts/init_model.py --config configs/model/murzik_15b_pilot.json --out /workspace/models/murzik-15b
  python scripts/init_model.py --config configs/model/murzik_32b.json --out /workspace/models/murzik-32b-moe
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def copy_modeling_code(out_dir: Path, model_type: str = "murzik") -> None:
    from scripts.murzik_hf_export import export_murzik_code

    export_murzik_code(out_dir, model_type=model_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model config JSON")
    parser.add_argument("--out", required=True, help="Output HF model directory")
    parser.add_argument("--tokenizer", default=None, help="Path to .model SentencePiece file")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    import murzik  # noqa: F401 — register Auto classes
    from murzik.configuration_murzik import MurzikConfig
    from murzik.configuration_murzik_moe import MurzikMoeConfig

    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, encoding="utf-8") as f:
        cfg_dict = json.load(f)

    skip = {"architectures", "transformers_version", "torch_dtype", "auto_map"}
    cfg_kwargs = {k: v for k, v in cfg_dict.items() if k not in skip}
    if cfg_dict["model_type"] == "murzik_moe":
        config = MurzikMoeConfig(**cfg_kwargs)
    else:
        config = MurzikConfig(**cfg_kwargs)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.bfloat16)

    copy_modeling_code(out_dir, model_type=cfg_dict["model_type"])
    model.save_pretrained(out_dir, safe_serialization=True)
    config.save_pretrained(out_dir)

    from scripts.murzik_hf_export import auto_map_for

    auto_map = auto_map_for(cfg_dict["model_type"])

    tok_path = Path(args.tokenizer) if args.tokenizer else Path("/workspace/data/tokenizer/murzik-spm128k.model")
    if not tok_path.is_file():
        raise SystemExit(
            f"SentencePiece model not found: {tok_path}\n"
            "Run: python scripts/train_tokenizer.py --out /workspace/data/tokenizer/murzik-spm128k.model"
        )
    from murzik.tokenization_murzik import MurzikTokenizer

    dest_model = out_dir / "murzik.model"
    shutil.copy2(tok_path, dest_model)
    tokenizer = MurzikTokenizer(vocab_file=str(dest_model))
    tokenizer.save_pretrained(out_dir)

    cfg_saved = out_dir / "config.json"
    with open(cfg_saved, encoding="utf-8") as f:
        saved = json.load(f)
    saved["auto_map"] = auto_map
    with open(cfg_saved, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved {out_dir}")
    print(f"  model_type: {cfg_dict['model_type']}")
    print(f"  parameters: {n_params / 1e9:.2f}B")


if __name__ == "__main__":
    main()
