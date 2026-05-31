#!/usr/bin/env python3
"""Post-PT smoke QA: greedy/sample generation on identity + multilingual prompts."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SYSTEM_PROMPT = (
    "You are Murzik, a multilingual language model developed by NULLXES. "
    "Answer clearly and concisely in the user's language."
)

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
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use murzik chat template (required for SFT checkpoints)",
    )
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
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

    for fname in ("modeling_murzik.py", "configuration_murzik.py", "tokenization_murzik.py"):
        src = ROOT / "murzik" / fname
        dst = ckpt / fname
        if src.is_file():
            shutil.copy2(src, dst)
    shutil.copytree(ROOT / "murzik", ckpt / "murzik", dirs_exist_ok=True)

    print(f"=== Murzik {'chat' if args.chat else 'PT'} smoke QA ===")
    print(f"time: {datetime.now(timezone.utc).isoformat()}")
    print(f"checkpoint: {ckpt}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        str(ckpt), trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    for lang, prompt in PROMPTS:
        if args.chat:
            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": prompt},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            inputs = {"input_ids": input_ids}
            prompt_len = input_ids.shape[-1]
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[-1]

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": False,
        }
        if args.chat:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(
                do_sample=True,
                temperature=args.temperature,
                top_p=0.9,
            )

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        generated = out[0][prompt_len:]
        continuation = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"[{lang}] PROMPT: {prompt!r}")
        print(f"[{lang}] CONT: {continuation.strip()!r}")
        print("---")

    print("=== done ===")


if __name__ == "__main__":
    main()
