#!/usr/bin/env python3
"""QA for identity SFT checkpoints (murzik chat format, fixed tokenizer ids)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SYSTEM = "You are Murzik, a multilingual language model developed by NULLXES."
QUESTIONS = [
    "Who are you?",
    "Кто ты?",
    "What is NULLXES MURZIK?",
]


def sync_repo_code(ckpt: Path) -> None:
    shutil.copytree(ROOT / "murzik", ckpt / "murzik", dirs_exist_ok=True)
    for fname in ("modeling_murzik.py", "configuration_murzik.py", "tokenization_murzik.py"):
        src = ROOT / "murzik" / fname
        if src.is_file():
            shutil.copy2(src, ckpt / fname)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/workspace/checkpoints/sft-15b-identity-hard",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    import murzik  # noqa: F401
    from murzik.tokenization_murzik import MurzikTokenizer

    ckpt = Path(args.checkpoint)
    sync_repo_code(ckpt)

    spm = ckpt / "murzik.model"
    if not spm.is_file():
        spm = Path("/workspace/data/tokenizer/murzik-spm128k.model")

    tok = MurzikTokenizer(vocab_file=str(spm))
    embed_rows = None  # filled after model load
    print(f"checkpoint: {ckpt}")
    print(f"len(tokenizer)={len(tok)} spm={tok.vocab_size} bos/eos={tok.bos_token_id}/{tok.eos_token_id}")
    for piece in ("<|user|>", "<|assistant|>", "<|system|>"):
        print(f"  {piece} -> {tok.sp_model.piece_to_id(piece)}")

    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    embed_rows = model.get_input_embeddings().weight.shape[0]
    print(f"config.vocab_size={model.config.vocab_size} embed_rows={embed_rows}")
    if embed_rows != model.config.vocab_size:
        print("WARNING: embedding rows != config.vocab_size — SFT likely needs rerun with fixed tokenizer")

    for q in QUESTIONS:
        ids = tok.encode_murzik_prompt(SYSTEM, q)
        input_ids = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        ans = tok.decode(out[0][len(ids) :].tolist(), skip_special_tokens=False)
        ans = ans.split("<|end|>")[0].strip()
        print(f"Q: {q}")
        print(f"A: {ans}")
        print("---")


if __name__ == "__main__":
    main()
