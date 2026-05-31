# LlamaFactory + кастомная модель MURZIK

Пошаговая инструкция: как подключить **MurzikMoE** к [LlamaFactory](https://github.com/hiyouga/LlamaFactory) и подготовить датасеты.

---

## Схема

```mermaid
flowchart TB
    A[murzik/ HF model code] --> B[scripts/init_model.py]
    B --> C[/workspace/models/murzik-15b]
    D[data/raw] --> E[scripts/prepare_dataset.py]
    E --> F[/workspace/data/dataset_info.json]
    G[llamafactory_ext/register_murzik.py] --> H[template murzik]
    C --> I[llamafactory-cli train]
    F --> I
    H --> I
```

---

## 1. Установка на RunPod (без Docker)

```bash
bash /workspace/NULLXES_MURZIK/scripts/setup_runpod.sh
```

Или вручную:

```bash
cd /workspace
git clone https://github.com/MagistrTheOne/NULLXES_MURZIK.git
cd NULLXES_MURZIK

pip install -r runpod/requirements.txt
pip install git+https://github.com/hiyouga/LlamaFactory.git

export PYTHONPATH=/workspace/NULLXES_MURZIK:$PYTHONPATH
```

---

## 2. Кастомная модель — что нужно LlamaFactory

LlamaFactory грузит модель через Hugging Face API. Для кастомной архитектуры нужны **3 вещи**:

| # | Что | Где в репо |
|---|-----|------------|
| 1 | Код модели (`trust_remote_code`) | `murzik/` |
| 2 | HF-чекпоинт (config + weights + auto_map) | `scripts/init_model.py` |
| 3 | Chat template `murzik` | `llamafactory_ext/register_murzik.py` |

### 2.1 Экспорт модели (random init)

**Dense pilot 15B (1× H200):**

```bash
python scripts/init_model.py \
  --config configs/model/murzik_15b_pilot.json \
  --out /workspace/models/murzik-15b \
  --tokenizer /workspace/data/tokenizer/murzik-spm128k.model
```

**MoE 32B:**

```bash
python scripts/init_model.py \
  --config configs/model/murzik_32b.json \
  --out /workspace/models/murzik-32b-moe
```

В `config.json` появится:

```json
"auto_map": {
  "AutoConfig": "murzik.configuration_murzik.MurzikConfig",
  "AutoModelForCausalLM": "murzik.modeling_murzik.MurzikForCausalLM"
}
```

### 2.2 Регистрация template (обязательно для SFT/DPO)

LlamaFactory **не знает** template `murzik` из коробки. Перед train:

```bash
python llamafactory_ext/register_murzik.py
```

Или через wrapper (рекомендуется):

```bash
bash scripts/llamafactory_train.sh configs/training/sft_murzik_15b_pilot.yaml
```

Формат чата:

```
<|murzik|><|system|>
{system}<|end|>
<|user|>
{user}<|end|>
<|assistant|>
{assistant}<|end|>
```

### 2.3 MoE + DeepSpeed (32B+)

Скрипт `register_murzik.py` добавляет `MurzikSparseMoeBlock` в ZeRO-3 leaf modules LlamaFactory.

В YAML всегда:

```yaml
trust_remote_code: true
bf16: true
moe_aux_loss_coef: 0.001   # для MoE
deepspeed: configs/training/ds_zero3_moe.json
gradient_checkpointing: true
```

---

## 3. Подготовка датасета

LlamaFactory читает **`dataset_info.json`** из `dataset_dir` (по умолчанию `./data`).

Структура на Volume:

```
/workspace/data/
├── dataset_info.json      ← реестр датасетов
├── examples/              ← демо для smoke test
├── pt/murzik_pt.jsonl     ← pretrain
├── sft/murzik_sft.json    ← chat SFT
└── dpo/murzik_dpo.json    ← preferences
```

### 3.1 Pre-training (stage: pt)

**Формат файла** — JSONL, одна колонка `text`:

```json
{"text": "длинный документ без chat-обёртки..."}
```

**dataset_info.json:**

```json
"murzik_pt": {
  "file_name": "pt/murzik_pt.jsonl",
  "columns": { "prompt": "text" }
}
```

**YAML:**

```yaml
stage: pt
dataset: murzik_pt
dataset_dir: /workspace/data
template: default        # НЕ murzik — для PT шаблон чата не нужен
cutoff_len: 4096
packing: true
```

### 3.2 SFT (stage: sft)

**Формат ShareGPT** (`conversations`):

```json
{
  "system": "You are MURZIK...",
  "conversations": [
    {"from": "human", "value": "вопрос"},
    {"from": "gpt", "value": "ответ"}
  ]
}
```

**dataset_info.json:**

```json
"murzik_sft": {
  "file_name": "sft/murzik_sft.json",
  "formatting": "sharegpt",
  "columns": { "messages": "conversations", "system": "system" },
  "tags": {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "system_tag": "system"
  }
}
```

**YAML:**

```yaml
stage: sft
dataset: murzik_sft
dataset_dir: /workspace/data
template: murzik          ← после register_murzik.py
cutoff_len: 8192
```

### 3.3 DPO (stage: dpo)

```json
{
  "conversations": [{"from": "human", "value": "..."}],
  "chosen": {"from": "gpt", "value": "хороший ответ"},
  "rejected": {"from": "gpt", "value": "плохой ответ"}
}
```

```json
"murzik_dpo": {
  "file_name": "dpo/murzik_dpo.json",
  "ranking": true,
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### 3.4 Автоконвертация из сырых файлов

```bash
# PT из .txt / .jsonl
python scripts/prepare_dataset.py \
  --out-dir /workspace/data \
  --pt /workspace/raw/corpus.txt

# SFT из JSON
python scripts/prepare_dataset.py \
  --out-dir /workspace/data \
  --sft /workspace/raw/dialogs.json

# DPO
python scripts/prepare_dataset.py \
  --out-dir /workspace/data \
  --dpo /workspace/raw/preferences.json

# Только демо-примеры (smoke test)
python scripts/prepare_dataset.py --out-dir /workspace/data --copy-examples
cp data/dataset_info.json /workspace/data/dataset_info.json
```

---

## 4. Запуск обучения

### Smoke test (15B pilot, demo data)

```bash
# 1) модель
python scripts/init_model.py \
  --config configs/model/murzik_15b_pilot.json \
  --out /workspace/models/murzik-15b

# 2) данные
python scripts/prepare_dataset.py --out-dir /workspace/data --copy-examples

# 3) PT
bash scripts/llamafactory_train.sh configs/training/pt_murzik_15b_pilot.yaml

# 4) SFT
bash scripts/llamafactory_train.sh configs/training/sft_murzik_15b_pilot.yaml
```

### Полный PT 32B MoE (кластер)

```bash
python scripts/init_model.py \
  --config configs/model/murzik_32b.json \
  --out /workspace/models/murzik-32b-moe

bash scripts/runpod/launch_pt_cluster.sh configs/training/pt_murzik_32b.yaml
```

---

## 5. Чеклист перед train

- [ ] `dataset_info.json` лежит в `dataset_dir`
- [ ] `model_name_or_path` — **папка** HF-модели, не голый JSON
- [ ] `trust_remote_code: true`
- [ ] Для SFT/DPO: `python llamafactory_ext/register_murzik.py`
- [ ] Special tokens в tokenizer совпадают с template (`<|user|>`, `<|end|>`, …)
- [ ] PT использует `template: default`, SFT — `template: murzik`

---

## 6. Частые ошибки

| Ошибка | Решение |
|--------|---------|
| `Template murzik does not exist` | Запустить `register_murzik.py` или wrapper script |
| `dataset murzik_pt not found` | Проверить `dataset_info.json` + путь `file_name` |
| OOM на 15B | `gradient_checkpointing: true`, ZeRO-3, `cutoff_len: 2048` |
| `trust_remote_code` | Добавить в YAML; код модели должен быть в папке чекпоинта |
| Пустой loss на SFT | Проверить ShareGPT: `human`/`gpt` в нечётных/чётных позициях |

---

## 7. Ссылки

- [LlamaFactory data README](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md)
- [LlamaFactory custom model docs](https://llamafactory.readthedocs.io/en/latest/advanced/model_support.html)
- Репозиторий: [MagistrTheOne/NULLXES_MURZIK](https://github.com/MagistrTheOne/NULLXES_MURZIK)
