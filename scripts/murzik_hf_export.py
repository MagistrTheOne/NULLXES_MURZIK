"""Helpers to export Murzik checkpoints compatible with Transformers 5.x trust_remote_code."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Transformers 5.x dynamic modules expect "module.Class" (single dot), not nested paths.
_STUBS = {
    "configuration_murzik.py": "from murzik.configuration_murzik import MurzikConfig\n",
    "modeling_murzik.py": "from murzik.modeling_murzik import MurzikForCausalLM\n",
    "tokenization_murzik.py": "from murzik.tokenization_murzik import MurzikTokenizer\n",
    "configuration_murzik_moe.py": "from murzik.configuration_murzik_moe import MurzikMoeConfig\n",
    "modeling_murzik_moe.py": "from murzik.modeling_murzik_moe import MurzikMoeForCausalLM\n",
}


def auto_map_for(model_type: str) -> dict[str, str]:
    if model_type == "murzik_moe":
        return {
            "AutoConfig": "configuration_murzik_moe.MurzikMoeConfig",
            "AutoModelForCausalLM": "modeling_murzik_moe.MurzikMoeForCausalLM",
            "AutoTokenizer": "tokenization_murzik.MurzikTokenizer",
        }
    return {
        "AutoConfig": "configuration_murzik.MurzikConfig",
        "AutoModelForCausalLM": "modeling_murzik.MurzikForCausalLM",
        "AutoTokenizer": "tokenization_murzik.MurzikTokenizer",
    }


def export_murzik_code(model_dir: Path, model_type: str = "murzik") -> None:
    """Copy murzik package + root stub modules for HF dynamic loading."""
    shutil.copytree(ROOT / "murzik", model_dir / "murzik", dirs_exist_ok=True)
    stub_names = (
        (
            "configuration_murzik_moe.py",
            "modeling_murzik_moe.py",
            "tokenization_murzik.py",
        )
        if model_type == "murzik_moe"
        else (
            "configuration_murzik.py",
            "modeling_murzik.py",
            "tokenization_murzik.py",
        )
    )
    for name in stub_names:
        (model_dir / name).write_text(_STUBS[name], encoding="utf-8")
