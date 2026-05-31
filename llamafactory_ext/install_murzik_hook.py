#!/usr/bin/env python3
"""Patch LlamaFactory launcher so Murzik template registers in every torchrun worker."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "# NULLXES MURZIK template register"
INJECT = f'''{MARKER}
import sys as _murzik_sys
_murzik_root = {str(ROOT)!r}
if _murzik_root not in _murzik_sys.path:
    _murzik_sys.path.insert(0, _murzik_root)
try:
    from llamafactory.data.template import TEMPLATES as _MURZIK_T
    if "murzik" not in _MURZIK_T:
        from llamafactory_ext.register_murzik import register_murzik_moe_patch, register_murzik_template
        register_murzik_template()
        register_murzik_moe_patch()
except Exception as _murzik_e:
    print("[murzik] launcher hook:", _murzik_e, file=_murzik_sys.stderr)

'''


def find_launcher() -> Path:
    try:
        import llamafactory

        return Path(llamafactory.__file__).parent / "launcher.py"
    except ImportError as exc:
        raise SystemExit("LlamaFactory not installed") from exc


def main() -> None:
    launcher = find_launcher()
    text = launcher.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[murzik] launcher already patched: {launcher}")
        return
    launcher.write_text(INJECT + text, encoding="utf-8")
    print(f"[murzik] launcher patched: {launcher}")


if __name__ == "__main__":
    main()
