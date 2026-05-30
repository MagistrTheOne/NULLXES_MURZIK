# NULLXES MURZIK — Training Pipeline

Training runs **only on RunPod**. **LlamaFactory** is the unified trainer for pre-training (PT), supervised fine-tuning (SFT), and DPO.

---

## 1. Phase overview

| Phase | Stage | LlamaFactory `stage` | Hardware (32B) |
|-------|-------|----------------------|----------------|
| 0 | Tokenizer | external (SPM) | 1× CPU pod |
| 0.5 | 1B dense pilot | `pt` | 1× H100 80GB |
| 1 | Pretrain 32B | `pt` | 4–8× H100 Instant Cluster |
| 2 | SFT chat | `sft` | 2–4× H100 |
| 3 | DPO align | `dpo` | 2–4× H100 |
| 4 | Scale 64B/100B | `pt` → `sft` | 8× H100 cluster |

---

## 2. LlamaFactory integration

### 2.1 Custom model registration (required)

After implementing `murzik_moe` in Transformers:

1. **`src/llamafactory/extras/constants.py`** — register model group + template
2. **`src/llamafactory/data/template.py`** — `register_template` for `murzik`
3. **`src/llamafactory/model/model_utils/moe.py`** — add `MurzikMoeSparseBlock` as ZeRO-3 leaf module
4. Pin fork in RunPod Docker image (see `runpod/Dockerfile`)

### 2.2 MoE training settings (all PT/SFT)

```yaml
# Always set for MoE
bf16: true
moe_aux_loss_coef: 0.001          # fallback; primary balance = bias updater
deepspeed: configs/training/ds_zero3_moe.json
gradient_checkpointing: true
```

**Do not** use fp16 for router/gate weights—bf16 or fp32 for routing stability.

### 2.3 LoRA strategy (SFT budget mode)

If full SFT is too expensive:

```yaml
finetuning_type: lora
lora_rank: 64
lora_target: q_proj,k_proj,v_proj,o_proj
# Experts: use lora_parameters (PEFT ≥0.17) for shared experts first
lora_parameters: shared_experts.*,gate.*
moe_aux_loss_coef: 0.01
```

Order of impact: **shared experts > attention > routed experts**.

---

## 3. Pre-training (32B)

**Config:** `configs/training/pt_murzik_32b.yaml`

### Hyperparameters (starting point)

| Param | Value |
|-------|-------|
| Global batch | 4M tokens/step (scale with cluster) |
| Seq length | 4096 (PT); 8192 optional phase 2 |
| LR | 3e-4 peak, cosine decay |
| Warmup | 2K steps |
| Weight decay | 0.1 |
| Adam β | (0.9, 0.95) |
| Tokens | 2T minimum for 32B |
| Init | std=0.006 |

### Data format (PT)

LlamaFactory JSONL:

```json
{"text": "raw document text without chat wrappers"}
```

Shards: `data/pt/shard-{0000..}.jsonl` on Network Volume.

---

## 4. Supervised fine-tuning

**Config:** `configs/training/sft_murzik_32b.yaml`

| Param | Value |
|-------|-------|
| LR | 5e-6 – 1e-5 |
| Epochs | 2–3 |
| Seq length | 8192 |
| Template | `murzik` |
| Packing | true (if supported for MoE in your LF version) |

Dataset schema (ShareGPT-style):

```json
{
  "conversations": [
    {"from": "system", "value": "..."},
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

Map `human`/`gpt` → user/assistant in dataset converter script.

---

## 5. DPO

**Config:** `configs/training/dpo_murzik_32b.yaml`

| Param | Value |
|-------|-------|
| Beta | 0.1 |
| LR | 5e-7 |
| Reference model | SFT checkpoint (frozen) |

---

## 6. DeepSpeed ZeRO-3 + Expert Parallel

**Config:** `configs/training/ds_zero3_moe.json`

For MoE at 32B+:

- **ZeRO-3** shards optimizer states and params
- **Expert Parallel (EP)** size = number of GPUs per node (typically 8)
- Register MoE block as leaf module (no incorrect partition of experts)

Launch pattern on Instant Cluster (each node):

```bash
export NCCL_SOCKET_IFNAME=ens1
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun \
  --nnodes=${NUM_NODES} \
  --nproc_per_node=${NUM_TRAINERS} \
  --node_rank=${NODE_RANK} \
  --master_addr=${PRIMARY_ADDR} \
  --master_port=${PRIMARY_PORT} \
  $(which llamafactory-cli) train configs/training/pt_murzik_32b.yaml
```

Or use `scripts/runpod/launch_pt_cluster.sh`.

---

## 7. Checkpoint policy

| Interval | Action |
|----------|--------|
| Every 500 steps | Async save to `/workspace/checkpoints/` (Network Volume) |
| Daily | Snapshot copy to cold storage (S3/R2 via rclone) |
| Best SFT | Keep top-3 by eval loss + MT-Bench |

Resume:

```yaml
resume_from_checkpoint: /workspace/checkpoints/pt-32b-step-120000
```

---

## 8. 1B dense pilot (validate before 32B)

Purpose: validate **data pipeline**, **LlamaFactory PT**, and **loss stability** at 1/30th cost.

| Param | Value |
|-------|-------|
| Layers | 24 |
| Hidden | 1536 |
| Heads | 16 / KV 4 |
| FFN | dense only (no MoE) |
| Tokens | 50B |
| Hardware | 1× H100 |

Success criteria:

- Loss ↓ smooth over 50B tokens
- No NaN in 72h run
- Throughput documented tokens/sec/GPU

Then enable MoE in a **3B MoE pilot** (optional) before full 32B.

---

## 9. Monitoring

Log every step:

- `loss`, `lm_loss`, `moe_aux_loss`
- `expert_load/max`, `expert_load/min`, `dead_experts`
- `grad_norm`, `lr`
- `tokens_per_sec`, `mfu` (estimated)

Alert if:

- `dead_experts > 5%` for 1K steps
- `grad_norm` spikes > 10× median

---

## 10. Dependency pins (RunPod image)

```
torch>=2.4.0
transformers>=4.52.0
deepspeed>=0.16.0
llamafactory @ git+https://github.com/NULLXES/llamafactory-murzik.git
flash-attn>=2.6.0
wandb
sentencepiece
```

Maintain `runpod/requirements.txt` with exact hashes after first green run.
