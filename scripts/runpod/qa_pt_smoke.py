#!/usr/bin/env python3
"""Post-PT smoke QA: greedy/sample generation on identity + multilingual prompts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROMPTS = [
    ("en", "Who are you?"),
    ("en", "What is NULLXES MURZIK?"),
    ("en", "Who developed the Murzik language model?"),
    ("ru", "Кто ты?"),
    ("ru", "Что такое NULLXES MURZIK?"),
    ("de", "Was ist NULLXES MURZIK?"),
    ("fr", "Qu'est-ce que NULLXES MURZIK?"),
    ("es", "¿Qué es NULLXES MURZIK?"),
    ("en", "The capital of France is"),
    ("ru", "Столица России — это"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/workspace/checkpoints/pt-15b-multilingual-2x/checkpoint-1500",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import murzik  # noqa: F401

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    if not (ckpt / "config.json").is_file():
        nested = sorted(ckpt.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[-1]))
        if nested:
            ckpt = nested[-1]
        else:
            raise SystemExit(f"No config.json in {args.checkpoint}")

    print(f"=== Murzik PT smoke QA ===")
    print(f"time: {datetime.now(timezone.utc).isoformat()}")
    print(f"checkpoint: {ckpt}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    for lang, prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        continuation = text[len(prompt) :] if text.startswith(prompt) else text
        print(f"[{lang}] PROMPT: {prompt!r}")
        print(f"[{lang}] FULL: {text}")
        print(f"[{lang}] CONT: {continuation.strip()!r}")
        print("---")

    print("=== done ===")


if __name__ == "__main__":
    main()
