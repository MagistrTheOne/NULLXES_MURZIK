# MURZIK-15B — Foundation pre-training (postmortem + plan)

Analysis of the first RunPod PT run ([`murzik-15b-init`](https://huggingface.co/MagistrTheOne/murzik-15b-init), 1500 steps, wiki + identity) and the corrected NULLXES foundation pipeline.

---

## 1. What happened (why "каша")

| Factor | First run | 2026 expectation (13B dense) |
|--------|-----------|------------------------------|
| **Tokens seen** | ~49M | **260B+** (Chinchilla ~20× params) |
| **Fraction of budget** | **0.02%** | 100% |
| **Data** | HF Wikipedia samples (7 langs) | Diverse foundation mix (owned corpus) |
| **Identity share** | ~60 lines ×20 ≈ 1200 rows vs ~63k wiki articles | Identity upsampled in **owned** shards, not drowned by external hub |
| **Eval mode** | Chat-style questions on **PT** checkpoint | Completion / CE ladder; chat only after SFT |
| **Optimizer** | `adamw_bnb_8bit` | Full PT: `adamw_torch`, bf16 |
| **Init std** | `initializer_range: 0.02` | **0.006** (match MoE line + stable deep stacks) |

**Conclusion:** The model did not "break" — it was stopped at ~0.02% of the token budget on a narrow encyclopedic slice. Gibberish at init→1500 steps on 13B from scratch is **expected**. Wikipedia-only PT also yields repetitive, non-agentic text even with more tokens.

The HF checkpoint is a **pipeline smoke test**, not a deployable foundation model.

---

## 2. Root causes (ranked)

1. **Token starvation** — largest single issue. No amount of SFT fixes a base that has not learned language structure.
2. **Wrong data contract** — PT needs raw documents; chat QA on PT is misleading.
3. **External hub as "foundation"** — wiki/c4 are fine for **stack validation**, not for NULLXES agent foundation (license, mix, branding).
4. **Tokenizer trained on wiki only** — vocab biased toward encyclopedic English; under-represents NULLXES domains and dialogue prose.
5. **Hyperparameters** — 8-bit Adam + high init std + no packing on a budget run: acceptable for smoke, not for production PT.

---

## 3. NULLXES foundation data (owned corpus)

**Rule:** Production PT reads **only** `pt/murzik_pt.jsonl` built from NULLXES shards on the RunPod volume. No Aya/C4/wiki in the foundation config.

### Target mix (tune by eval; starting point)

| Shard | Share | Purpose |
|-------|-------|---------|
| `nullxes_docs` | 25% | Product, internal docs, policies |
| `nullxes_technical` | 20% | Code, APIs, configs, logs (sanitized) |
| `nullxes_dialogue_prose` | 20% | Conversational prose (not chat JSON) — agent tone |
| `nullxes_reasoning` | 15% | Explanations, step-by-step analysis |
| `nullxes_multilingual` | 15% | RU/EN/DE/… owned translations |
| `murzik_identity` | 5% | Branding (upsampled 50× in builder) |

Example **format** lives in `data/examples/` — replace with real NULLXES crawls, exports, and licensed text.

### Build on RunPod

```bash
cd /workspace/NULLXES_MURZIK
python scripts/build_foundation_corpus.py \
  --manifest data/foundation_manifest.json \
  --out-dir /workspace/data

python scripts/validate_pt_readiness.py --data-dir /workspace/data
```

---

## 4. Training stages (15B dense)

```
[1] Tokenizer  ← sample from merged foundation corpus (not wiki-only)
[2] init_model.py  ← initializer_range 0.006, fresh random weights
[3] PT foundation  ← configs/training/pt_murzik_15b_foundation.yaml
[4] QA ladder    ← scripts/runpod/qa_checkpoint_ladder.py (completion, not chat)
[5] SFT          ← identity + owned instruct data only
[6] DPO (opt)    ← preference pairs from NULLXES
[7] HF export    ← murzik-15b-init updated in place
```

### Token budget targets

| Milestone | Tokens | Expected behavior |
|-----------|--------|-------------------|
| Smoke | 1–5B | Loss ↓, no NaN; still gibberish completions |
| MVP base | 30–50B | Coherent sentences, basic facts |
| Agent-ready base | **200B+** | Stable multilingual completion, reasoning prose |
| Chinchilla-optimal | ~260B | Diminishing returns; eval-driven stop |

Compute steps (2× H200, batch≈32k tok/step): **~6M steps** for 200B tokens — plan cluster time accordingly.

### Config: `pt_murzik_15b_foundation.yaml`

- Dataset: `murzik_pt` only (local JSONL)
- `learning_rate: 2.0e-4`, `adamw_torch`, `warmup_ratio: 0.01`
- `cutoff_len: 4096`, `packing: true`
- `max_grad_norm: 1.0`, `weight_decay: 0.1`, β=(0.9, 0.95)
- DeepSpeed ZeRO-3 when multi-GPU

**Deprecated for foundation:** `pt_murzik_15b_fast_2x.yaml`, `*_multilingual*.yaml` — wiki hub smoke only.

---

## 5. Deploy readiness checklist

Before uploading to HF or wiring agents:

- [ ] `validate_pt_readiness.py` passes (corpus size, identity ratio, tokenizer on corpus)
- [ ] Train loss well below `ln(vocab) ≈ 11.76` and falling smoothly over 10k+ steps
- [ ] `qa_checkpoint_ladder.py`: PT stage gives **sensible completions** on factual prompts
- [ ] SFT identity hard pass (`qa_sft_identity.py`)
- [ ] Model card stage field updated (`hf_model_card/`)

---

## 6. First run → next run

| Action | Detail |
|--------|--------|
| **Do not** SFT the current 49M-token PT as product base | Insufficient PT |
| **Do** keep checkpoint for ladder/debug | Compare init vs failed-short-PT |
| **Re-init** weights with `initializer_range: 0.006` | New `init_model.py` run |
| **Retrain tokenizer** on foundation sample | `train_tokenizer.py --corpus /workspace/data/pt/murzik_pt.jsonl` |
| **Fill** `/workspace/data/pt/shards/*.jsonl` with NULLXES data | Examples in repo |
| **Run** foundation PT to token milestone before SFT | See §4 |

---

## 7. References

- Repo: [NULLXES_MURZIK](https://github.com/MagistrTheOne/NULLXES_MURZIK)
- Weights: [murzik-15b-init](https://huggingface.co/MagistrTheOne/murzik-15b-init)
- Architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
- RunPod ops: [RUNPOD.md](./RUNPOD.md)
