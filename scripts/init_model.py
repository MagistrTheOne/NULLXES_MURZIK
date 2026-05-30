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


def copy_modeling_code(out_dir: Path) -> None:
    murzik_src = ROOT / "murzik"
    out_murzik = out_dir / "murzik"
    if out_murzik.exists():
        shutil.rmtree(out_murzik)
    shutil.copytree(murzik_src, out_murzik)


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

    copy_modeling_code(out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    config.save_pretrained(out_dir)

    # trust_remote_code entrypoint
    auto_map = {
        "AutoConfig": "murzik.configuration_murzik_moe.MurzikMoeConfig"
        if cfg_dict["model_type"] == "murzik_moe"
        else "murzik.configuration_murzik.MurzikConfig",
        "AutoModelForCausalLM": "murzik.modeling_murzik_moe.MurzikMoeForCausalLM"
        if cfg_dict["model_type"] == "murzik_moe"
        else "murzik.modeling_murzik.MurzikForCausalLM",
    }
    if args.tokenizer:
        tok_path = Path(args.tokenizer)
        from murzik.tokenization_murzik import MurzikTokenizer

        tokenizer = MurzikTokenizer(vocab_file=str(tok_path))
        tokenizer.save_pretrained(out_dir)
        auto_map["AutoTokenizer"] = "murzik.tokenization_murzik.MurzikTokenizer"
    else:
        # Minimal tokenizer stub for PT-only smoke tests
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast(
            tokenizer_object=None,
            bos_token="<|murzik|>",
            eos_token="<|end|>",
            pad_token="<|pad|>",
            unk_token="<|unk|>",
        )
        tok.save_pretrained(out_dir)

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
