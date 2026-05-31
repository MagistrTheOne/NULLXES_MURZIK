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
base_model: MagistrTheOne/murzik-15b-init
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

# MURZIK-15B-PT-PILOT

**NULLXES MURZIK** — dense 15B pilot after multilingual pre-training smoke run.

| | |
|---|---|
| **Organization** | [NULLXES](https://nullxes.com) |
| **Base weights** | [MagistrTheOne/murzik-15b-init](https://huggingface.co/MagistrTheOne/murzik-15b-init) |
| **Architecture** | `MurzikForCausalLM` (custom) |
| **Training** | LlamaFactory full PT, 2× H200, 1500 steps |
| **Data** | Wikipedia (en/ru/de/es/fr/zh/uk) + Murzik identity corpus |
| **Seq length** | 2048 |
| **Precision** | bf16 |
| **Status** | Pilot checkpoint — not chat-tuned |

## Notes

This is a **pipeline validation** checkpoint (~0.37 epoch, ~49M tokens).  
Use for regression tests and continued PT/SFT — not production chat.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "MagistrTheOne/murzik-15b-pt-pilot"

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
