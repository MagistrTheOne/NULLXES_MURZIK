#!/usr/bin/env python3
"""
Merge NULLXES PT shards into pt/murzik_pt.jsonl for LlamaFactory.

Reads shard JSONL from data/pt/shards/ (repo) or /workspace/data/pt/shards/ (RunPod).
Uses foundation_manifest.json weights; identity shard upsampled via identity_repeat.

Usage:
  python scripts/seed_corpus_base.py --scale 2
  python scripts/build_foundation_corpus.py --out-dir /workspace/data
  python scripts/build_foundation_corpus.py --manifest data/foundation_manifest_language_core.json
  python scripts/build_foundation_corpus.py --out-dir /workspace/data --copy-shards
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
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


def resolve_shard(rel: str, root: Path, out_dir: Path) -> Path | None:
    for base in (out_dir, root):
        candidate = base / rel
        if candidate.is_file():
            return candidate.resolve()
    return None


def pick_shares(manifest: dict, root: Path, out_dir: Path) -> dict[str, float]:
    raw = manifest.get("shares") or {}
    shares: dict[str, float] = {}
    for rel, weight in raw.items():
        if rel.startswith("_"):
            continue
        path = resolve_shard(rel, root, out_dir)
        if path is not None:
            shares[str(path)] = float(weight)
        else:
            print(f"[warn] missing shard: {rel}")
    if not shares:
        raise SystemExit(
            "No shard files found. Run: python scripts/seed_corpus_base.py\n"
            "Or copy JSONL into /workspace/data/pt/shards/"
        )
    total = sum(shares.values())
    return {k: v / total for k, v in shares.items()}


def build_pool(
    shares: dict[str, float],
    min_chars: int,
    max_chars: int,
    identity_repeat: int,
    expand_target: int | None,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    docs_by_path: dict[str, list[str]] = {}

    for path_str in shares:
        path = Path(path_str)
        docs = load_jsonl(path, min_chars, max_chars)
        if "identity" in path.name.lower():
            docs = docs * identity_repeat
            print(f"[identity] {path.name}: {len(docs)} rows ({identity_repeat}×)")
        docs_by_path[path_str] = docs
        print(f"[load] {path.name}: {len(docs)} docs")

    raw_total = sum(len(d) for d in docs_by_path.values())
    pool: list[str] = []
    for path_str, share in shares.items():
        docs = docs_by_path[path_str]
        if not docs:
            continue
        n = max(1, int(round(share * raw_total)))
        if len(docs) >= n:
            chosen = rng.sample(docs, n)
        else:
            chosen = rng.choices(docs, k=n)
        pool.extend(chosen)
        print(f"[mix] {Path(path_str).name}: share={share:.1%} -> {len(chosen)} docs")

    if expand_target and len(pool) < expand_target:
        print(f"[expand] {len(pool)} -> {expand_target} docs (cycle for multi-epoch pilot)")
        base = list(pool)
        rng.shuffle(base)
        while len(pool) < expand_target:
            pool.extend(base)
        pool = pool[:expand_target]

    rng.shuffle(pool)
    return pool


def copy_shards(root: Path, out_dir: Path) -> None:
    src = root / "data" / "pt" / "shards"
    dst = out_dir / "pt" / "shards"
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.jsonl"):
        shutil.copy2(f, dst / f.name)
    print(f"[copy] {src} -> {dst}")


def write_corpus(out_dir: Path, pool: list[str]) -> Path:
    rel = Path("pt/murzik_pt.jsonl")
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for text in pool:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return rel


def write_stats(out_dir: Path, pool: list[str], manifest: dict, shares: dict[str, float]) -> None:
    char_total = sum(len(t) for t in pool)
    stats = {
        "documents": len(pool),
        "chars": char_total,
        "est_tokens": char_total // 4,
        "manifest": manifest.get("description"),
        "model_target": manifest.get("model_target"),
        "shares": {Path(k).name: v for k, v in shares.items()},
    }
    stats_path = out_dir / "pt" / "corpus_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[stats] docs={len(pool)} ~tokens={stats['est_tokens']:,} -> {stats_path}")


def write_dataset_info(out_dir: Path, pt_rel: str) -> None:
    info = {
        "murzik_pt": {
            "file_name": pt_rel.replace("\\", "/"),
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
    parser.add_argument("--copy-shards", action="store_true", help="Copy repo pt/shards into out-dir")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        manifest_path = root / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_shards or not (out_dir / "pt" / "shards").exists():
        copy_shards(root, out_dir)

    shares = pick_shares(manifest, root, out_dir)
    expand = manifest.get("expand_target_docs")
    expand_target = int(expand) if expand else None

    pool = build_pool(
        shares,
        min_chars=int(manifest.get("min_chars", 80)),
        max_chars=int(manifest.get("max_chars", 32_000)),
        identity_repeat=int(manifest.get("identity_repeat", 50)),
        expand_target=expand_target,
        seed=int(manifest.get("seed", 42)),
    )

    pt_rel = write_corpus(out_dir, pool)
    write_stats(out_dir, pool, manifest, shares)
    write_dataset_info(out_dir, str(pt_rel))
    print(f"[done] {out_dir / pt_rel}")


if __name__ == "__main__":
    main()
