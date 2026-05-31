---
language:
  - en
  - ru
  - de
  - es
  - fr
  - zh
  - uk
license: other
tags:
  - murzik
  - nullxes
  - causal-lm
  - custom_code
  - multilingual
  - pretraining
library_name: transformers
pipeline_tag: text-generation
---

# MURZIK-15B (PT pilot)

**NULLXES MURZIK** — custom causal language model (dense pilot, ~13B parameters).  
Multilingual **pre-training pilot** checkpoint (pipeline validation on RunPod).

| | |
|---|---|
| **Organization** | [NULLXES](https://nullxes.com) |
| **Contact** | [ceo@nullxes.com](mailto:ceo@nullxes.com) |
| **Architecture** | `MurzikForCausalLM` (custom, not a fork) |
| **Total params** | ~13B |
| **Context (train)** | 2048 |
| **Precision** | bf16 |
| **Status** | PT pilot — **not chat-tuned** (SFT next) |

## Training run (pilot)

| | |
|---|---|
| Steps | 1500 |
| GPUs | 2× H200 |
| Data | Wikipedia (en/ru/de/es/fr/zh/uk) + Murzik identity corpus |
| Tokens seen | ~49M (~0.37 epoch) |
| Avg train loss | ~61.3 |

This checkpoint validates tokenizer, data mix, LlamaFactory PT, and weight export.  
It is **not** intended for production chat until SFT with the `murzik` template.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "MagistrTheOne/murzik-15b-init"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
```

## Chat template (SFT — not applied in this PT checkpoint)

Template name in LlamaFactory: `murzik`

```
<|murzik|><|system|>
{system}<|end|>
<|user|>
{user}<|end|>
<|assistant|>
{assistant}<|end|>
```

## Roadmap

| Stage | Status |
|-------|--------|
| Random init | done |
| PT pilot | done (this repo) |
| SFT (identity + Aya) | planned |
| MoE 32B | Stage 2 |

## License

Proprietary — **NULLXES**. Weights are published for research and integration testing.  
Commercial use requires written permission: **ceo@nullxes.com**.

## Citation

```bibtex
@misc{murzik15b_init2026,
  title        = {NULLXES MURZIK-15B},
  author       = {NULLXES},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/MagistrTheOne/murzik-15b-init}},
  contact      = {ceo@nullxes.com}
}
```
