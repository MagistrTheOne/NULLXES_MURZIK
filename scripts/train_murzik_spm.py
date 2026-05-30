#!/usr/bin/env python3
"""Train Murzik SentencePiece tokenizer and attach it to an HF checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _collect_corpus(num_samples: int, hf_home: Path) -> Path:
    import os

    os.environ.setdefault("HF_HOME", str(hf_home))
    from datasets import load_dataset

    corpus = Path(tempfile.mkdtemp()) / "murzik_spm_corpus.txt"
    per_lang = max(num_samples // 2, 1)
    total = 0
    with corpus.open("w", encoding="utf-8") as out:
        for subset in ("20231101.en", "20231101.ru"):
            count = 0
            ds = load_dataset("wikimedia/wikipedia", subset, split="train", streaming=True)
            for row in ds:
                text = (row.get("text") or "").strip()
                if len(text) < 80:
                    continue
                out.write(text.replace("\n", " ") + "\n")
                count += 1
                total += 1
                if count >= per_lang:
                    break
    if total < 1000:
        raise RuntimeError(f"Corpus too small ({total} lines) for SPM training")
    print(f"[train_murzik_spm] Corpus lines: {total}")
    return corpus


def train_spm(model_dir: Path, vocab_size: int, num_samples: int, hf_home: Path) -> None:
    import sentencepiece as spm

    from murzik.tokenization_murzik import SPECIAL_TOKENS

    sys.path.insert(0, str(ROOT / "scripts"))
    from export_hf_remote_code import export_remote_code

    model_dir.mkdir(parents=True, exist_ok=True)
    corpus = _collect_corpus(num_samples=num_samples, hf_home=hf_home)
    prefix = model_dir / "murzik"
    user_symbols = SPECIAL_TOKENS["additional_special_tokens"]

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=0.9995,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=user_symbols,
        unk_piece=SPECIAL_TOKENS["unk_token"],
        bos_piece=SPECIAL_TOKENS["bos_token"],
        eos_piece=SPECIAL_TOKENS["eos_token"],
        pad_piece=SPECIAL_TOKENS["pad_token"],
    )

    from murzik.tokenization_murzik import MurzikTokenizer

    tokenizer = MurzikTokenizer(vocab_file=f"{prefix}.model")
    tokenizer.save_pretrained(model_dir)

    export_remote_code(model_dir)

    sample = tokenizer.encode("NULLXES MURZIK multilingual pretraining.", add_special_tokens=False)
    if len(sample) < 4:
        raise RuntimeError("SPM tokenizer produced too few tokens on sanity check")
    print(f"[train_murzik_spm] Saved MurzikTokenizer to {model_dir} ({len(sample)} tok sanity)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/murzik-15b")
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--hf-home", default="/workspace/cache/huggingface")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    vocab_size = args.vocab_size
    if vocab_size is None:
        with open(model_dir / "config.json", encoding="utf-8") as f:
            cfg_vocab = json.load(f).get("vocab_size", 128256)
        # Pilot SPM: 32k is enough for PT smoke; model embeddings stay at cfg size.
        vocab_size = min(cfg_vocab, 32768)

    train_spm(model_dir, vocab_size=vocab_size, num_samples=args.num_samples, hf_home=Path(args.hf_home))


if __name__ == "__main__":
    main()
