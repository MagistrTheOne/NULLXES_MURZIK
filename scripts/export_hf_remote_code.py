#!/usr/bin/env python3
"""Export Transformers-5-compatible flat remote-code files into an HF model directory."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MURZIK_SRC = ROOT / "murzik"

IMPORT_REPLACEMENTS = (
    (r"from \.configuration_murzik import", "from configuration_murzik import"),
    (r"from \.configuration_murzik_moe import", "from configuration_murzik_moe import"),
    (r"from \.modeling_murzik import", "from modeling_murzik import"),
)

FLAT_FILES = (
    "configuration_murzik.py",
    "configuration_murzik_moe.py",
    "modeling_murzik.py",
    "modeling_murzik_moe.py",
    "tokenization_murzik.py",
)


def _write_flat_file(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    for pattern, repl in IMPORT_REPLACEMENTS:
        text = re.sub(pattern, repl, text)
    dst.write_text(text, encoding="utf-8")


def export_remote_code(model_dir: Path, model_type: str | None = None) -> None:
    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    model_type = model_type or cfg.get("model_type", "murzik")
    is_moe = model_type == "murzik_moe"

    for name in FLAT_FILES:
        src = MURZIK_SRC / name
        if not src.exists():
            continue
        _write_flat_file(src, model_dir / name)

    auto_map = {
        "AutoConfig": "configuration_murzik_moe.MurzikMoeConfig"
        if is_moe
        else "configuration_murzik.MurzikConfig",
        "AutoModelForCausalLM": "modeling_murzik_moe.MurzikMoeForCausalLM"
        if is_moe
        else "modeling_murzik.MurzikForCausalLM",
    }
    if (model_dir / "murzik.model").exists() or (model_dir / "tokenizer_config.json").exists():
        auto_map["AutoTokenizer"] = "tokenization_murzik.MurzikTokenizer"

    cfg["auto_map"] = auto_map
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    # Keep package copy for local PYTHONPATH registration (optional).
    out_pkg = model_dir / "murzik"
    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    shutil.copytree(MURZIK_SRC, out_pkg)

    print(f"[export_hf_remote_code] Patched {model_dir}")
    print(f"  auto_map: {auto_map}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="HF checkpoint directory")
    parser.add_argument("--model-type", default=None, choices=["murzik", "murzik_moe"])
    args = parser.parse_args()
    export_remote_code(Path(args.model_dir), args.model_type)
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    main()
