#!/usr/bin/env python3
"""Compare SFT train prompt tokenization vs inference (run on RunPod)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import murzik  # noqa: F401
from murzik.tokenization_murzik import MURZIK_CHAT_TEMPLATE, MurzikTokenizer


def main() -> None:
    ckpt = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/checkpoints/sft-15b-identity-hard")
    spm = ckpt / "murzik.model"
    if not spm.is_file():
        spm = Path("/workspace/data/tokenizer/murzik-spm128k.model")
    tok = MurzikTokenizer(vocab_file=str(spm))

    print("=== MurzikTokenizer ===")
    print(f"spm pieces: {tok.sp_model.get_piece_size()}")
    print(f"len(tokenizer): {len(tok)}")
    print(f"bos/eos: {tok.bos_token_id} / {tok.eos_token_id}")

    for piece in ("<|murzik|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"):
        enc = tok.encode(piece, add_special_tokens=False)
        pid = tok.sp_model.piece_to_id(piece)
        print(f"  {piece}: encode={enc} spm_id={pid}")

    system = "You are Murzik, a multilingual language model developed by NULLXES."
    user = "Who are you?"
    answer = "I am Murzik, a language model developed by NULLXES."

    manual = (
        f"<|murzik|><|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user}<|end|>\n<|assistant|>\n{answer}<|end|>"
    )
    ids_manual = tok.encode(manual, add_special_tokens=False)
    print(f"\nmanual full seq len={len(ids_manual)}")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    tmpl = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids_tmpl = tok.encode(tmpl, add_special_tokens=False)
    print(f"chat_template prompt len={len(ids_tmpl)}")
    print(f"manual == template: {manual == tmpl}")
    if manual != tmpl:
        print("--- manual ---")
        print(repr(manual))
        print("--- template ---")
        print(repr(tmpl))

    sft_path = Path("/workspace/data/sft/murzik_sft.json")
    if sft_path.is_file():
        rows = json.loads(sft_path.read_text(encoding="utf-8"))
        ex = rows[0]
        print("\n=== train example[0] ===")
        print("system:", ex.get("system", "")[:80])
        print("Q:", ex["conversations"][0]["value"])
        print("A:", ex["conversations"][1]["value"][:80])

    log = Path("/workspace/logs/sft-hard.log")
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        for needle in ("template", "murzik", "Cannot find", "Template"):
            if needle.lower() in text.lower():
                pass
        for line in text.splitlines():
            if "template" in line.lower() or "murzik" in line.lower():
                print("log:", line[:200])

    # LlamaFactory-style encode (if available)
    try:
        from llamafactory.data.template import get_template_and_fix_tokenizer

        t = get_template_and_fix_tokenizer(tok, name="murzik")
        train_ids, train_labels = t.encode_oneturn(
            tokenizer=tok,
            messages=[{"role": "user", "content": user}],
            system=system,
        )
        print(f"\nLlamaFactory murzik encode_oneturn len={len(train_ids)}")
        print("first tokens:", tok.convert_ids_to_tokens(train_ids[:25]))
        print("matches manual prefix:", train_ids[:15] == ids_manual[:15])
    except Exception as exc:
        print(f"\nLlamaFactory encode skipped: {exc}")

    print(f"\nchat_template in tokenizer: {tok.chat_template is not None}")
    print(f"matches MURZIK_CHAT_TEMPLATE: {tok.chat_template == MURZIK_CHAT_TEMPLATE}")


if __name__ == "__main__":
    main()
