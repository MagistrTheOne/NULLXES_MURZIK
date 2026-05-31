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
Canonical Hugging Face repo: **[MagistrTheOne/murzik-15b-init](https://huggingface.co/MagistrTheOne/murzik-15b-init)**

| | |
|---|---|
| **Organization** | [NULLXES](https://nullxes.com) |
| **Contact** | [ceo@nullxes.com](mailto:ceo@nullxes.com) |
| **Architecture** | `murzik` (custom, not a fork) |
| **Total params** | ~13B |
| **Context** | 8192 (config); PT run used 2048 |
| **Precision** | bf16 |

See the live model card on Hugging Face for the current training stage and usage.

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

## License

Proprietary — **NULLXES**. Contact: **ceo@nullxes.com**.
