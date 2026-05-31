#!/usr/bin/env python3
"""
Pre-flight checks before Murzik foundation PT on RunPod.

Usage:
  python scripts/validate_pt_readiness.py --data-dir /workspace/data
  python scripts/validate_pt_readiness.py --data-dir /workspace/data --min-docs 10000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"WARN: {msg}")


def ok(msg: str) -> None:
    print(f"OK: {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NULLXES PT readiness")
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--min-docs", type=int, default=500, help="Minimum documents in murzik_pt.jsonl")
    parser.add_argument("--min-est-tokens", type=int, default=10_000_000, help="Warn below this token estimate")
    parser.add_argument("--model-config", default="configs/model/murzik_15b_pilot.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = Path(args.data_dir)
    pt_file = data / "pt" / "murzik_pt.jsonl"
    info_file = data / "dataset_info.json"
    tok_file = data / "tokenizer" / "murzik-spm128k.model"

    if not pt_file.is_file():
        fail(f"Missing {pt_file}. Run: python scripts/build_foundation_corpus.py --out-dir {data}")

    lines = [l for l in pt_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) < args.min_docs:
        warn(f"Only {len(lines)} documents (min {args.min_docs}). PT will not produce deployable base.")

    chars = 0
    identity_hits = 0
    for line in lines:
        obj = json.loads(line)
        text = obj.get("text", "")
        chars += len(text)
        if re_search_nullxes(text):
            identity_hits += 1

    est_tokens = chars // 4
    ok(f"Corpus: {len(lines)} docs, ~{est_tokens:,} tokens (char/4 estimate)")
    if est_tokens < args.min_est_tokens:
        warn(
            f"~{est_tokens:,} tokens << 200B target for agent-ready 15B. "
            "Expect gibberish until token budget is met."
        )

    ratio = identity_hits / max(len(lines), 1)
    if ratio < 0.01:
        warn(f"Low NULLXES/Murzik mention rate ({ratio:.1%}). Check identity upsampling.")
    else:
        ok(f"Identity signal in {ratio:.1%} of documents")

    if not info_file.is_file():
        warn(f"Missing {info_file}. build_foundation_corpus.py writes it automatically.")
    else:
        info = json.loads(info_file.read_text(encoding="utf-8"))
        if "murzik_pt" not in info:
            fail("dataset_info.json missing murzik_pt entry")
        ok("dataset_info.json has murzik_pt")

    cfg_path = root / args.model_config
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        init = cfg.get("initializer_range", 0.02)
        if init > 0.01:
            warn(f"initializer_range={init} — use 0.006 for new init_model.py runs")
        else:
            ok(f"initializer_range={init}")

    if not tok_file.is_file():
        warn(f"Tokenizer missing: {tok_file}. Train on foundation sample before init.")
    else:
        ok(f"Tokenizer present: {tok_file}")

    stats_file = data / "pt" / "corpus_stats.json"
    if stats_file.is_file():
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        ok(f"Stats: {stats_file}")

    random_ce = math.log(128256)
    print()
    print(f"Reference: random-init CE ≈ {random_ce:.2f}. PT loss should fall well below this.")
    print("Eval: use completion prompts (qa_checkpoint_ladder.py), not chat, before SFT.")
    print("All critical paths checked.")


def re_search_nullxes(text: str) -> bool:
    lower = text.lower()
    return "nullxes" in lower or "murzik" in lower


if __name__ == "__main__":
    main()
