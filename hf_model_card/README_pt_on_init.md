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
library_name: transformers
pipeline_tag: text-generation
---

# MURZIK-15B-INIT

**NULLXES MURZIK** — custom causal language model (dense ~13B).  
Canonical Hugging Face repo for the Murzik-15B dense line.

| | |
|---|---|
| **Organization** | [NULLXES](https://nullxes.com) |
| **Contact** | [ceo@nullxes.com](mailto:ceo@nullxes.com) |
| **Architecture** | `MurzikForCausalLM` (custom, not a fork) |
| **Total params** | ~13B |
| **Precision** | bf16 |
| **HF repo** | `MagistrTheOne/murzik-15b-init` (this page) |

## Current checkpoint

| | |
|---|---|
| **Stage** | Pre-training (first run — **smoke only**, not deployable) |
| **Steps** | 1500 |
| **Data** | HF Wikipedia samples + tiny identity (see `docs/FOUNDATION_PT.md`) |
| **Tokens seen** | ~49M (**0.02%** of Chinchilla budget for 13B) |
| **Next** | Re-init + NULLXES foundation PT (`pt_murzik_15b_foundation.yaml`) |

Weights in this repo are **updated in place** (random init → PT → later SFT).  
The repo name stays **`murzik-15b-init`**; only the README and files change per stage.

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

## Chat template (after SFT)

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
| Pre-training | smoke (1500 steps, insufficient — see FOUNDATION_PT.md) |
| Foundation PT (NULLXES corpus) | **next** |
| MoE 32B | separate line |

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
