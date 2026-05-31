"""Murzik dense decoder (pilot). GQA + RoPE + SwiGLU + RMSNorm."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.generation import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from .configuration_murzik import MurzikConfig

logger = logging.get_logger(__name__)


class MurzikRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MurzikRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(max_position_embeddings, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class MurzikMLP(nn.Module):
    def __init__(self, config: MurzikConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MurzikAttention(nn.Module):
    def __init__(self, config: MurzikConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = MurzikRMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else None
        self.k_norm = MurzikRMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else None
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past = (k, v) if use_cache else None

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output), past


class MurzikDecoderLayer(nn.Module):
    def __init__(self, config: MurzikConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MurzikAttention(config, layer_idx)
        self.mlp = MurzikMLP(config)
        self.input_layernorm = MurzikRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MurzikRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_embeddings, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states, attention_mask, position_embeddings, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present


class MurzikPreTrainedModel(PreTrainedModel):
    config_class = MurzikConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MurzikDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class MurzikModel(MurzikPreTrainedModel):
    def __init__(self, config: MurzikConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([MurzikDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
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
        for idx, layer in enumerate(self.layers):
            past = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                hidden_states, present = self._checkpoint_layer(
                    layer, hidden_states, attention_mask, position_embeddings, past, use_cache
                )
            else:
                hidden_states, present = layer(
                    hidden_states, attention_mask, position_embeddings, past, use_cache
                )
            if use_cache:
                presents.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents

    def _checkpoint_layer(self, layer, hidden_states, attention_mask, position_embeddings, past, use_cache):
        def custom_forward(hs):
            out, pr = layer(hs, attention_mask, position_embeddings, past, use_cache)
            return out, pr

        return torch.utils.checkpoint.checkpoint(custom_forward, hidden_states, use_reentrant=False)


class MurzikForCausalLM(MurzikPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MurzikConfig):
        super().__init__(config)
        self.model = MurzikModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.generation_config = GenerationConfig.from_model_config(config)

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
        hidden_states, past_key_values = self.model(
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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )
