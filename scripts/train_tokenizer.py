#!/usr/bin/env python3
"""Train Murzik SentencePiece tokenizer from cached Wikipedia samples."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from murzik.tokenization_murzik import SPECIAL_TOKENS  # noqa: E402

WIKI_SUBSETS = [
    ("wikimedia/wikipedia", "20231101.en", 30_000),
    ("wikimedia/wikipedia", "20231101.ru", 20_000),
    ("wikimedia/wikipedia", "20231101.de", 10_000),
    ("wikimedia/wikipedia", "20231101.es", 10_000),
    ("wikimedia/wikipedia", "20231101.fr", 10_000),
]


def export_corpus(corpus_path: Path, max_samples: int) -> int:
    from datasets import load_dataset

    lines = 0
    with corpus_path.open("w", encoding="utf-8") as out:
        for hub, subset, cap in WIKI_SUBSETS:
            if lines >= max_samples:
                break
            print(f"[tokenizer] sampling {subset} (cap {cap})...")
            ds = load_dataset(hub, subset, split="train", trust_remote_code=False)
            take = min(cap, max_samples - lines, len(ds))
            for i in range(take):
                text = ds[i].get("text", "").strip()
                if len(text) < 80:
                    continue
                out.write(text.replace("\n", " ") + "\n")
                lines += 1
                if lines >= max_samples:
                    break
    return lines


def train_spm(corpus_path: Path, prefix: Path, vocab_size: int) -> None:
    symbols = SPECIAL_TOKENS["additional_special_tokens"] + [
        SPECIAL_TOKENS["pad_token"],
        SPECIAL_TOKENS["bos_token"],
        SPECIAL_TOKENS["eos_token"],
        SPECIAL_TOKENS["unk_token"],
    ]
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        hard_vocab_limit=False,
        user_defined_symbols=",".join(symbols),
        num_threads=16,
    )
    print(f"[tokenizer] trained {prefix}.model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Murzik SPM tokenizer")
    parser.add_argument(
        "--out",
        default="/workspace/data/tokenizer/murzik-spm128k.model",
        help="Output .model path",
    )
    parser.add_argument("--vocab-size", type=int, default=128256)
    parser.add_argument("--max-samples", type=int, default=80_000)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        corpus = tmp_dir / "corpus.txt"
        prefix = tmp_dir / "murzik-spm"
        n = export_corpus(corpus, args.max_samples)
        if n < 1000:
            raise RuntimeError(f"Corpus too small ({n} lines). Check HF cache / network.")
        print(f"[tokenizer] corpus lines: {n}")
        train_spm(corpus, prefix, args.vocab_size)
        shutil.copy2(f"{prefix}.model", out_path)
        vocab_out = out_path.with_suffix(".vocab")
        shutil.copy2(f"{prefix}.vocab", vocab_out)

    print(f"[tokenizer] saved {out_path}")


if __name__ == "__main__":
    main()
