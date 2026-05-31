# NULLXES Murzik — pre-training corpus base

Canonical shard directory committed in git. **Not** copied from Llama, Qwen, DeepSeek, or Wikipedia dumps.

## Model target

| Field | Value |
|-------|--------|
| Architecture | `MurzikForCausalLM` (custom `model_type: murzik`) |
| Config | `configs/model/murzik_15b_pilot.json` |
| Params | ~13B dense (15B line) |
| HF repo | [MagistrTheOne/murzik-15b-init](https://huggingface.co/MagistrTheOne/murzik-15b-init) |
| MoE line | Separate — `murzik_32b.json` / `murzik_moe` (later) |

**Mix design** follows 2026 foundation practice (general prose + technical + multilingual + reasoning),  
but every document is **original NULLXES text** for Murzik agents.

## Shards

| File | Tier | Role |
|------|------|------|
| `nullxes_language.jsonl` | A | General prose — **learn language first** |
| `nullxes_docs.jsonl` | A | NULLXES product / platform docs |
| `nullxes_technical.jsonl` | B | Code, APIs, infra |
| `nullxes_dialogue_prose.jsonl` | B | Conversational narrative (not chat JSON) |
| `nullxes_reasoning.jsonl` | B | Analysis, checklists, math prose |
| `nullxes_multilingual.jsonl` | A | RU/EN/DE/FR/ES/UK/ZH snippets |
| `murzik_identity.jsonl` | branding | Upsampled 30–50× in builder (~5% share) |

## Customize

1. **Add lines** to any `*.jsonl` — one JSON object per line: `{"text": "..."}`
2. **Add new shard** — create file here, add path + weight to `data/foundation_manifest.json`
3. **Regenerate seed base** (optional): `python scripts/seed_corpus_base.py --scale 2`
4. **Rebuild merged corpus**:

```bash
python scripts/build_foundation_corpus.py \
  --manifest data/foundation_manifest_language_core.json \
  --out-dir /workspace/data \
  --copy-shards
```

## Manifests

| File | When |
|------|------|
| `foundation_manifest_language_core.json` | **Stage 1** — language first (expand to 50k docs for pilot epochs) |
| `foundation_manifest.json` | **Stage 2** — full mix after core checkpoint |

## 2× H200 one-liner

```bash
cd /workspace/NULLXES_MURZIK
python scripts/seed_corpus_base.py --scale 2          # optional refresh
bash scripts/runpod/bootstrap_foundation_pt.sh \
  configs/training/pt_murzik_15b_2x.yaml
```

Edit `bootstrap_foundation_pt.sh` manifest line for stage 2 full mix.

See also [docs/FOUNDATION_PT.md](../docs/FOUNDATION_PT.md).
