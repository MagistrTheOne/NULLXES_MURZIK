# Multilingual training (MURZIK-15B)

## Selected Hugging Face datasets

| Key | HF repo | Stage | Languages | Notes |
|-----|---------|-------|-----------|-------|
| `wiki_*` | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | PT | en, ru, de, es, fr, zh, uk | `num_samples` capped per lang for pilot |
| `mc4_en`, `mc4_ru` | [mc4](https://huggingface.co/datasets/mc4) | PT | EN, RU | Web crawl, good diversity |
| `aya_sft` | [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | SFT | **65 langs** | 204k human-annotated pairs |

LlamaFactory loads HF datasets directly via `hf_hub_url` in `dataset_info.json` — no manual JSONL export needed.

## Setup on RunPod

```bash
cd /workspace/NULLXES_MURZIK
git pull
bash scripts/setup_multilingual_datasets.sh /workspace/data
python llamafactory_ext/register_murzik.py
```

## Train

```bash
# PT multilingual mix
bash scripts/llamafactory_train.sh configs/training/pt_murzik_15b_multilingual.yaml

# SFT Aya multilingual
bash scripts/llamafactory_train.sh configs/training/sft_murzik_15b_multilingual.yaml
```

## Adjust language mix

Edit `data/dataset_info_multilingual.json`:

- Change `num_samples` per language
- Add subsets: `20231101.ja`, `20231101.ar`, etc.
- PT YAML `dataset:` line — comma-separated keys

## Model source

Public weights: [MagistrTheOne/murzik-15b-init](https://huggingface.co/MagistrTheOne/murzik-15b-init)

```yaml
model_name_or_path: MagistrTheOne/murzik-15b-init
trust_remote_code: true
```
