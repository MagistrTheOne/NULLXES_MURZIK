#!/usr/bin/env python3
"""Compare init → PT → SFT on completion prompts (RunPod diagnostic)."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_STAGES = [
    ("init", "/workspace/models/murzik-15b"),
    ("pt", "/workspace/checkpoints/pt-15b-multilingual-2x"),
    ("sft", "/workspace/checkpoints/sft-15b-identity-hard"),
]

COMPLETION_PROMPTS = [
    "The capital of France is",
    "Столица России — это",
    "Murzik was developed by",
    "NULLXES MURZIK is",
    "1 + 1 =",
]

CHAT_QUESTIONS = [
    "Who are you?",
    "What is NULLXES MURZIK?",
]


def sync_code(ckpt: Path) -> None:
    if not ckpt.is_dir():
        return
    shutil.copytree(ROOT / "murzik", ckpt / "murzik", dirs_exist_ok=True)
    for fname in ("modeling_murzik.py", "configuration_murzik.py", "tokenization_murzik.py"):
        src = ROOT / "murzik" / fname
        if src.is_file():
            shutil.copy2(src, ckpt / fname)


def resolve_ckpt(path: Path) -> Path | None:
    if not path.is_dir():
        return None
    if (path / "config.json").is_file() and (
        (path / "model.safetensors").is_file() or list(path.glob("*.safetensors"))
    ):
        return path
    nested = sorted(path.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    if nested and (nested[-1] / "config.json").is_file():
        return nested[-1]
    return path if (path / "config.json").is_file() else None


def find_spm(ckpt: Path) -> Path:
    for candidate in (
        ckpt / "murzik.model",
        Path("/workspace/data/tokenizer/murzik-spm128k.model"),
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No murzik.model near {ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QA ladder: init / PT / SFT")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--chat", action="store_true", help="Also run murzik chat QA on SFT stage")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=[],
        help="Optional list name=path (default: init, pt, sft on /workspace)",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    import murzik  # noqa: F401
    from murzik.tokenization_murzik import MurzikTokenizer

    stages = DEFAULT_STAGES
    if args.stages:
        stages = []
        for item in args.stages:
            name, path = item.split("=", 1)
            stages.append((name, path))

    random_ce = math.log(128256)
    print("=== Murzik checkpoint ladder ===")
    print(f"theoretical random-init CE (ln vocab): {random_ce:.2f}")
    print(f"if train_loss >> {random_ce:.0f} early PT may be miscalibrated or still warming up")
    print()

    system = "You are Murzik, a multilingual language model developed by NULLXES."

    for name, raw in stages:
        ckpt = resolve_ckpt(Path(raw))
        print(f"{'=' * 60}")
        print(f"STAGE: {name}  path: {raw}")
        if ckpt is None:
            print("  SKIP: directory or weights missing")
            print()
            continue
        sync_code(ckpt)
        spm = find_spm(ckpt)
        tok = MurzikTokenizer(vocab_file=str(spm))

        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(ckpt),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
            )
        except Exception as exc:
            print(f"  SKIP load failed: {exc}")
            print()
            continue

        model.eval()
        embed = model.get_input_embeddings().weight.shape[0]
        print(f"  resolved: {ckpt}")
        print(f"  vocab={model.config.vocab_size} embed_rows={embed} len(tok)={len(tok)}")

        for prompt in COMPLETION_PROMPTS:
            ids = tok.encode(prompt, add_special_tokens=False)
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
            cont = tok.decode(out[0][len(ids) :].tolist(), skip_special_tokens=True)
            print(f"  [complete] {prompt!r}")
            print(f"             -> {cont[:200]!r}")

        if args.chat or name == "sft":
            for q in CHAT_QUESTIONS:
                ids = tok.encode_murzik_prompt(system, q)
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
                print(f"  [chat] Q: {q!r}")
                print(f"         A: {ans[:200]!r}")

        del model
        torch.cuda.empty_cache()
        print()

    print("=== done ===")
    print("Interpretation:")
    print("  init gibberish + PT gibberish  -> need much more PT (15B from scratch needs 100k+ steps)")
    print("  init gibberish + PT sensible   -> PT OK, re-run SFT with fixed tokenizer")
    print("  PT sensible + SFT identity bad -> re-run identity SFT only (~5 min)")


if __name__ == "__main__":
    main()
