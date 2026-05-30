#!/usr/bin/env python3
"""
Prepare NULLXES datasets for LlamaFactory.

PT:   raw .txt / .jsonl with "text" -> pt/*.jsonl
SFT:  JSON/CSV dialogs -> sft/murzik_sft.json (sharegpt)
DPO:  preference pairs -> dpo/murzik_dpo.json

Also writes data/dataset_info.json into --out-dir.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_pt(sources: list[Path], out_dir: Path, min_chars: int = 100) -> str:
    rows = []
    for src in sources:
        if src.suffix == ".jsonl":
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or ""
                if len(text) >= min_chars:
                    rows.append({"text": text.strip()})
        else:
            text = src.read_text(encoding="utf-8")
            for chunk in re.split(r"\n{2,}", text):
                chunk = chunk.strip()
                if len(chunk) >= min_chars:
                    rows.append({"text": chunk})

    rel = "pt/murzik_pt.jsonl"
    write_jsonl(out_dir / rel, rows)
    return rel


def prepare_sft(sources: list[Path], out_dir: Path) -> str:
    conversations = []
    for src in sources:
        data = json.loads(src.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        for item in data:
            if "conversations" in item:
                conversations.append(item)
            elif "messages" in item:
                conv = []
                for m in item["messages"]:
                    role = m.get("role", "user")
                    tag = {"user": "human", "assistant": "gpt", "system": "system"}.get(role, "human")
                    conv.append({"from": tag, "value": m["content"]})
                conversations.append({"conversations": conv, "system": item.get("system")})
            elif "instruction" in item:
                conv = [{"from": "human", "value": item["instruction"]}]
                if item.get("input"):
                    conv[0]["value"] += "\n" + item["input"]
                conv.append({"from": "gpt", "value": item["output"]})
                entry = {"conversations": conv}
                if item.get("system"):
                    entry["system"] = item["system"]
                conversations.append(entry)

    rel = "sft/murzik_sft.json"
    write_json(out_dir / rel, conversations)
    return rel


def prepare_dpo(sources: list[Path], out_dir: Path) -> str:
    rows = []
    for src in sources:
        data = json.loads(src.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        rows.extend(data)

    rel = "dpo/murzik_dpo.json"
    write_json(out_dir / rel, rows)
    return rel


def write_dataset_info(out_dir: Path, pt_file: str | None, sft_file: str | None, dpo_file: str | None) -> None:
    info = {}

    if pt_file:
        info["murzik_pt"] = {
            "file_name": pt_file,
            "columns": {"prompt": "text"},
        }
        info["murzik_pt_demo"] = {
            "file_name": "examples/pt_demo.jsonl",
            "columns": {"prompt": "text"},
        }

    if sft_file:
        info["murzik_sft"] = {
            "file_name": sft_file,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
            },
        }
        info["murzik_sft_demo"] = {
            "file_name": "examples/sft_demo.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
            },
        }

    if dpo_file:
        info["murzik_dpo"] = {
            "file_name": dpo_file,
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }

    write_json(out_dir / "dataset_info.json", info)
    print(f"Wrote {out_dir / 'dataset_info.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LlamaFactory datasets for MURZIK")
    parser.add_argument("--out-dir", default="/workspace/data", help="dataset_dir for LlamaFactory")
    parser.add_argument("--pt", nargs="*", default=[], help="Raw files for pre-training")
    parser.add_argument("--sft", nargs="*", default=[], help="SFT source JSON files")
    parser.add_argument("--dpo", nargs="*", default=[], help="DPO source JSON files")
    parser.add_argument("--copy-examples", action="store_true", help="Copy demo examples into out-dir")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)

    if args.copy_examples or not any([args.pt, args.sft, args.dpo]):
        examples = root / "data" / "examples"
        for name in ["pt_demo.jsonl", "sft_demo.json", "dpo_demo.json"]:
            src = examples / name
            if src.exists():
                dst = out_dir / "examples" / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    pt_file = prepare_pt([Path(p) for p in args.pt], out_dir) if args.pt else None
    sft_file = prepare_sft([Path(p) for p in args.sft], out_dir) if args.sft else None
    dpo_file = prepare_dpo([Path(p) for p in args.dpo], out_dir) if args.dpo else None

    write_dataset_info(out_dir, pt_file, sft_file, dpo_file)


if __name__ == "__main__":
    main()
