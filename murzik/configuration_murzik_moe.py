"""MurzikMoE config (32B/64B/100B)."""

from transformers import PretrainedConfig


class MurzikMoeConfig(PretrainedConfig):
    model_type = "murzik_moe"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2560,
        intermediate_size: int = 9728,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.006,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        rope_theta: float = 1_000_000.0,
        attention_dropout: float = 0.0,
        use_qk_norm: bool = True,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 2432,
        num_experts: int = 96,
        num_experts_per_tok: int = 6,
        num_shared_experts: int = 2,
        first_k_dense_replace: int = 2,
        router_aux_loss_coef: float = 0.001,
        expert_bias_update_speed: float = 0.001,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.first_k_dense_replace = first_k_dense_replace
        self.router_aux_loss_coef = router_aux_loss_coef
        self.expert_bias_update_speed = expert_bias_update_speed
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
