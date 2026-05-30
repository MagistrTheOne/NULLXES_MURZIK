# NULLXES MURZIK

Custom chat LLM family (30B–100B total parameters, MoE) trained from scratch on proprietary datasets. Training runs exclusively on **RunPod**; fine-tuning and alignment use **LlamaFactory**.

## Design goals

| Goal | Choice |
|------|--------|
| Not a fork | Original **MurzikMoE** architecture (Qwen/DeepSeek-inspired, legally and technically distinct) |
| Efficiency | Fine-grained MoE + shared experts; ~5–12B **active** params per token |
| Chat-first | Native chat template, SFT → DPO pipeline |
| Ops | RunPod Instant Clusters + Network Volumes; LlamaFactory YAML configs |
| Reproducibility | Pinned deps, W&B logging, checkpoint policy |

## Model lineup (target)

| Name | Total | Active/token | Context | Primary use |
|------|-------|--------------|---------|-------------|
| **MURZIK-32B** | ~32B | ~5B | 32K → 128K | MVP, iteration speed |
| **MURZIK-64B** | ~64B | ~8B | 32K → 128K | Production balance |
| **MURZIK-100B** | ~100B | ~12B | 32K → 128K | Maximum quality |

Start with **MURZIK-32B**; scale configs after tokenizer + 1B pilot complete.

## Repository layout

```
NULLXES_MURZIK/
├── docs/                 # Architecture, training, RunPod playbooks
├── configs/
│   ├── model/            # MurzikMoE HF config JSON (32B/64B/100B)
│   └── training/         # LlamaFactory YAML (pt, sft, dpo)
├── murzik/               # Custom HF model + tokenizer (Phase 2)
├── scripts/              # RunPod launchers, data prep
├── data/                 # .gitignore — datasets live on RunPod Volume
└── runpod/               # Docker template + cluster bootstrap
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — MurzikMoE blocks, MoE routing, parameter math
- [LlamaFactory setup](docs/LLAMAFACTORY_SETUP.md) — custom model, datasets, train commands
- [Training pipeline](docs/TRAINING.md) — phases, LlamaFactory, DeepSpeed
- [RunPod operations](docs/RUNPOD.md) — clusters, volumes, cost controls

## Quick start (after Phase 1 scaffold)

1. Mount RunPod Network Volume at `/workspace/data`.
2. Train tokenizer: `scripts/train_tokenizer.sh`
3. Pilot 1B dense on 1× H100 → validate data + stack.
4. Pretrain MURZIK-32B on Instant Cluster (see `docs/TRAINING.md`).
5. SFT with LlamaFactory: `configs/training/sft_murzik_32b.yaml`

## License

Proprietary — NULLXES. Weights and datasets are not public unless explicitly released.
