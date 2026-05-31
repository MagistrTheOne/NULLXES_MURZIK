#!/usr/bin/env python3
"""Install Python startup hook so Murzik template registers in every torchrun worker."""

from __future__ import annotations

import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "# NULLXES MURZIK LlamaFactory hook"
HOOK = f'''
{MARKER}
import os
if os.environ.get("MURZIK_REGISTER_TEMPLATE") == "1":
    import sys as _sys
    _root = {str(ROOT)!r}
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    try:
        from llamafactory.data.template import TEMPLATES
        if "murzik" not in TEMPLATES:
            from llamafactory_ext.register_murzik import (
                register_murzik_moe_patch,
                register_murzik_template,
            )
            register_murzik_template()
            register_murzik_moe_patch()
    except Exception as _exc:
        print(f"[murzik] hook failed: {{_exc}}", file=_sys.stderr)
'''


def main() -> None:
    site_dir = Path(site.getsitepackages()[0])
    target = site_dir / "sitecustomize.py"
    text = target.read_text(encoding="utf-8") if target.is_file() else ""
    if MARKER in text:
        print(f"[murzik] hook already installed: {target}")
        return
    target.write_text(text + HOOK, encoding="utf-8")
    print(f"[murzik] hook installed: {target}")
    print("  export MURZIK_REGISTER_TEMPLATE=1 before llamafactory-cli train")


if __name__ == "__main__":
    main()
