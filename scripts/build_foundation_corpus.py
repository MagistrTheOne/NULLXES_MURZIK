#!/usr/bin/env python3
"""
Merge NULLXES-owned PT shards into pt/murzik_pt.jsonl for LlamaFactory.

Uses weighted sampling from foundation_manifest.json. Identity shard is
upsampled via identity_repeat (default 50×) so branding survives broad corpora.

Usage:
  python scripts/build_foundation_corpus.py
  python scripts/build_foundation_corpus.py --manifest data/foundation_manifest.json --out-dir /workspace/data
  python scripts/build_foundation_corpus.py --production   # use production_shards paths
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def load_jsonl(path: Path, min_chars: int, max_chars: int) -> list[str]:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        text = (obj.get("text") or obj.get("content") or "").strip()
        if len(text) < min_chars:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        rows.append(text)
    return rows


def pick_shares(manifest: dict, production: bool, root: Path, out_dir: Path) -> dict[str, float]:
    key = "production_shards" if production else "shares"
    raw = manifest.get(key) or manifest["shares"]
    shares: dict[str, float] = {}
    for rel, weight in raw.items():
        if rel.startswith("_"):
            continue
        for base in (out_dir, root):
            candidate = base / rel
            if candidate.is_file():
                shares[str(candidate.resolve())] = float(weight)
                break
        else:
            candidate = root / rel
            if candidate.is_file():
                shares[str(candidate.resolve())] = float(weight)
    if not shares:
        raise SystemExit("No shard files found. Copy examples to /workspace/data/pt/shards/ or use repo examples.")
    total = sum(shares.values())
    return {k: v / total for k, v in shares.items()}


def build_pool(
    shares: dict[str, float],
    min_chars: int,
    max_chars: int,
    identity_repeat: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    pool: list[str] = []

    for path_str, share in shares.items():
        path = Path(path_str)
        docs = load_jsonl(path, min_chars, max_chars)
        if not docs:
            print(f"[warn] empty shard: {path}")
            continue
        if "identity" in path.name.lower():
            docs = docs * identity_repeat
            print(f"[identity] {path.name}: {len(docs)} rows after {identity_repeat}× repeat")
        # Target count proportional to share; at least one doc per shard
        target = max(1, int(round(share * 10_000)))
        if len(docs) >= target:
            chosen = rng.sample(docs, target) if target < len(docs) else docs
        else:
            chosen = [rng.choice(docs) for _ in range(target)]
        pool.extend(chosen)
        print(f"[shard] {path.name}: share={share:.2%} picked={len(chosen)}")

    rng.shuffle(pool)
    return pool


def write_corpus(out_dir: Path, pool: list[str]) -> Path:
    rel = Path("pt/murzik_pt.jsonl")
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for text in pool:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return rel


def write_stats(out_dir: Path, pool: list[str], shares: dict[str, float]) -> None:
    char_total = sum(len(t) for t in pool)
    est_tokens = char_total // 4
    stats = {
        "documents": len(pool),
        "chars": char_total,
        "est_tokens": est_tokens,
        "shares": shares,
    }
    stats_path = out_dir / "pt" / "corpus_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[stats] docs={len(pool)} ~tokens={est_tokens:,} -> {stats_path}")


def write_dataset_info(out_dir: Path, pt_rel: str) -> None:
    info = {
        "murzik_pt": {
            "file_name": pt_rel.replace("\\", "/"),
            "columns": {"prompt": "text"},
        },
        "murzik_identity": {
            "file_name": "pt/murzik_identity.jsonl",
            "columns": {"prompt": "text"},
        },
    }
    path = out_dir / "dataset_info.json"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[dataset_info] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NULLXES foundation PT corpus")
    parser.add_argument("--manifest", default="data/foundation_manifest.json")
    parser.add_argument("--out-dir", default="/workspace/data")
    parser.add_argument("--production", action="store_true", help="Use production_shards paths")
    parser.add_argument("--copy-examples", action="store_true", help="Copy repo examples into out-dir")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        manifest_path = root / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_examples or not (out_dir / "pt").exists():
        examples = root / "data" / "examples"
        for name in examples.glob("*.jsonl"):
            dst = out_dir / "examples" / name.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(name.read_text(encoding="utf-8"), encoding="utf-8")
        (out_dir / "pt" / "murzik_identity.jsonl").parent.mkdir(parents=True, exist_ok=True)
        id_src = examples / "murzik_identity.jsonl"
        if id_src.is_file():
            (out_dir / "pt" / "murzik_identity.jsonl").write_text(
                id_src.read_text(encoding="utf-8"), encoding="utf-8"
            )

    shares = pick_shares(manifest, args.production, root, out_dir)
    pool = build_pool(
        shares,
        min_chars=int(manifest.get("min_chars", 80)),
        max_chars=int(manifest.get("max_chars", 32_000)),
        identity_repeat=int(manifest.get("identity_repeat", 50)),
        seed=int(manifest.get("seed", 42)),
    )
    if len(pool) < 100:
        print("[warn] corpus very small — for smoke tests only; production needs millions of documents")

    pt_rel = write_corpus(out_dir, pool)
    write_stats(out_dir, pool, shares)
    write_dataset_info(out_dir, str(pt_rel))
    print(f"[done] {out_dir / pt_rel}")


if __name__ == "__main__":
    main()
