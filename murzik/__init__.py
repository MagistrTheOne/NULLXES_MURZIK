from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .configuration_murzik import MurzikConfig
from .configuration_murzik_moe import MurzikMoeConfig
from .modeling_murzik import MurzikForCausalLM
from .modeling_murzik_moe import MurzikMoeForCausalLM
from .tokenization_murzik import MurzikTokenizer

AutoConfig.register("murzik", MurzikConfig)
AutoConfig.register("murzik_moe", MurzikMoeConfig)
AutoModelForCausalLM.register(MurzikConfig, MurzikForCausalLM)
AutoModelForCausalLM.register(MurzikMoeConfig, MurzikMoeForCausalLM)
AutoTokenizer.register(MurzikConfig, slow_tokenizer_class=MurzikTokenizer)

__all__ = [
    "MurzikConfig",
    "MurzikMoeConfig",
    "MurzikForCausalLM",
    "MurzikMoeForCausalLM",
    "MurzikTokenizer",
]
