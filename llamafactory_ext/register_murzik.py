#!/usr/bin/env python3
"""Register Murzik chat template in LlamaFactory (call before train)."""

from __future__ import annotations

import sys


def register_murzik_template() -> None:
    try:
        from llamafactory.data.formatter import EmptyFormatter, StringFormatter
        from llamafactory.data.template import register_template
    except ImportError as exc:
        raise SystemExit(
            "LlamaFactory not installed. Run: pip install git+https://github.com/hiyouga/LlamaFactory.git"
        ) from exc

    # Murzik chat format:
    # <|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n{assistant}<|end|>
    register_template(
        name="murzik",
        format_user=StringFormatter(
            slots=["<|user|>\n", "{{content}}", "<|end|>\n", "<|assistant|>\n"]
        ),
        format_assistant=StringFormatter(slots=["{{content}}", "<|end|>", {"eos_token"}]),
        format_system=StringFormatter(slots=["<|system|>\n", "{{content}}", "<|end|>\n"]),
        format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
        stop_words=["<|end|>"],
        replace_eos=True,
        enable_thinking=False,
    )
    print("[murzik] Template 'murzik' registered in LlamaFactory.")


def register_murzik_moe_patch() -> None:
    """Optional: register MoE block as DeepSpeed ZeRO-3 leaf module."""
    try:
        from llamafactory.model.model_utils import moe as lf_moe
        from murzik.modeling_murzik_moe import MurzikSparseMoeBlock
    except ImportError:
        print("[murzik] Skip MoE leaf registration (LlamaFactory or murzik not on path).")
        return

    leaf = getattr(lf_moe, "_MOE_LEAF_MODULES", None)
    if leaf is not None and MurzikSparseMoeBlock not in leaf:
        leaf.append(MurzikSparseMoeBlock)
        print("[murzik] MurzikSparseMoeBlock added to ZeRO-3 leaf modules.")


if __name__ == "__main__":
    register_murzik_template()
    register_murzik_moe_patch()
