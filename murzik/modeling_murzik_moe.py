"""MurzikMoE — sparse MoE FFN on top of Murzik decoder blocks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_murzik_moe import MurzikMoeConfig
from .modeling_murzik import (
    MurzikAttention,
    MurzikMLP,
    MurzikPreTrainedModel,
    MurzikRMSNorm,
    MurzikRotaryEmbedding,
)


class MurzikMoeMLP(nn.Module):
    """Single expert SwiGLU block."""

    def __init__(self, config: MurzikMoeConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MurzikSparseMoeBlock(nn.Module):
    def __init__(self, config: MurzikMoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MurzikMoeMLP(config, config.moe_intermediate_size) for _ in range(config.num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [MurzikMoeMLP(config, config.moe_intermediate_size) for _ in range(config.num_shared_experts)]
        )
        self.register_buffer("expert_bias", torch.zeros(config.num_experts), persistent=True)
        self.router_aux_loss_coef = config.router_aux_loss_coef

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits + self.expert_bias, dim=-1, dtype=torch.float32)
        routing_weights, selected = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)

        out = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            mask = (selected == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            expert_input = flat[idx]
            expert_out = expert(expert_input)
            weight = (selected[idx] == expert_idx).float() * routing_weights[idx]
            weight = weight.sum(dim=-1, keepdim=True)
            out[idx] += expert_out * weight

        for shared in self.shared_experts:
            out += shared(flat)

        aux_loss = self._aux_loss(router_logits, selected)
        return out.view(batch_size, seq_len, hidden_dim), aux_loss

    def _aux_loss(self, router_logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        one_hot = F.one_hot(selected, num_classes=self.num_experts).float().sum(dim=1)
        load = one_hot.mean(dim=0)
        balance = probs.mean(dim=0)
        aux = self.num_experts * (load * balance).sum()
        return aux * self.router_aux_loss_coef


class MurzikMoeDecoderLayer(nn.Module):
    def __init__(self, config: MurzikMoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = MurzikAttention(config, layer_idx)
        self.input_layernorm = MurzikRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MurzikRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        use_moe = layer_idx >= config.first_k_dense_replace
        if use_moe:
            self.mlp = MurzikSparseMoeBlock(config)
            self.is_moe = True
        else:
            self.mlp = MurzikMLP(config)
            self.is_moe = False

    def forward(self, hidden_states, attention_mask, position_embeddings, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states, attention_mask, position_embeddings, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        aux_loss = None
        if self.is_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present, aux_loss


class MurzikMoePreTrainedModel(MurzikPreTrainedModel):
    config_class = MurzikMoeConfig
    _no_split_modules = ["MurzikMoeDecoderLayer"]


class MurzikMoeModel(MurzikMoePreTrainedModel):
    def __init__(self, config: MurzikMoeConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [MurzikMoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = MurzikRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MurzikRotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        position_embeddings = (cos, sin)

        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        presents = [] if use_cache else None
        aux_loss = torch.tensor(0.0, device=input_ids.device)
        for idx, layer in enumerate(self.layers):
            past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present, layer_aux = layer(
                hidden_states, attention_mask, position_embeddings, past, use_cache
            )
            if layer_aux is not None:
                aux_loss = aux_loss + layer_aux
            if use_cache:
                presents.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents, aux_loss


class MurzikMoeForCausalLM(MurzikMoePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MurzikMoeConfig):
        super().__init__(config)
        self.model = MurzikMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if aux_loss is not None:
                loss = loss + aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )
