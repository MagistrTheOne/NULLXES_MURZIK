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

**NULLXES MURZIK** — custom causal language model (dense pilot, ~13B parameters).  
Random-init weights for pre-training and multilingual fine-tuning on proprietary NULLXES pipelines.

| | |
|---|---|
| **Organization** | [NULLXES](https://nullxes.com) |
| **Contact** | [ceo@nullxes.com](mailto:ceo@nullxes.com) |
| **Architecture** | `murzik` (custom, not a fork) |
| **Total params** | ~13B |
| **Active params** | ~13B (dense pilot) |
| **Context** | 8K (base config) |
| **Precision** | bf16 |
| **Status** | Init checkpoint — PT / SFT in progress |

## Model lineage

- **Codename:** MurzikMoE family (this checkpoint: dense 15B pilot)
- **Training stack:** LlamaFactory + RunPod H200
- **Target lineup:** MURZIK-32B / 64B / 100B MoE (lower active params)

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

## Chat template (SFT)

Template name in LlamaFactory: `murzik`

```
<|murzik|><|system|>
{system}<|end|>
<|user|>
{user}<|end|>
<|assistant|>
{assistant}<|end|>
```

## Multilingual training (planned / in progress)

| Stage | Datasets |
|-------|----------|
| PT | Wikipedia (en, ru, de, es, fr, zh, uk), mC4 (en, ru) |
| SFT | [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) (65 languages) |

## License

Proprietary — **NULLXES**. Weights are published for research and integration testing.  
Commercial use requires written permission: **ceo@nullxes.com**.

## Citation

```bibtex
@misc{murzik15b_init2026,
  title        = {NULLXES MURZIK-15B Init Checkpoint},
  author       = {NULLXES},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/MagistrTheOne/murzik-15b-init}},
  contact      = {ceo@nullxes.com}
}
```
