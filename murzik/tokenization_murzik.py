"""Murzik tokenizer — SentencePiece wrapper for Hugging Face."""

from pathlib import Path
from typing import Optional

import sentencepiece as spm
from transformers import PreTrainedTokenizer

# Special tokens (must match SFT template)
SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "bos_token": "<|murzik|>",
    "eos_token": "<|end|>",
    "unk_token": "<|unk|>",
    "additional_special_tokens": [
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
    ],
}

# Must match llamafactory_ext/register_murzik.py (LlamaFactory template "murzik").
MURZIK_CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' -%}"
    "{{ '<|murzik|><|system|>\\n' + messages[0]['content'] + '<|end|>\\n' }}"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}"
    "{{ '<|murzik|>' }}"
    "{%- set loop_messages = messages -%}"
    "{%- endif -%}"
    "{%- for message in loop_messages -%}"
    "{%- if message['role'] == 'user' -%}"
    "{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{{ '<|assistant|>\\n' + message['content'] + '<|end|>' }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{ '<|assistant|>\\n' }}"
    "{%- endif -%}"
)


class MurzikTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "murzik.model"}
    model_input_names = ["input_ids", "attention_mask"]

    @staticmethod
    def _resolve_vocab_path(vocab_file: str | None, kwargs: dict) -> str | None:
        if not vocab_file:
            return None
        path = Path(vocab_file)
        if path.is_file():
            return str(path.resolve())
        for key in ("name_or_path", "_name_or_path"):
            root = kwargs.get(key)
            if root:
                candidate = Path(root) / vocab_file
                if candidate.is_file():
                    return str(candidate.resolve())
        return str(vocab_file)

    def __init__(
        self,
        vocab_file: str | None = None,
        bos_token: str = SPECIAL_TOKENS["bos_token"],
        eos_token: str = SPECIAL_TOKENS["eos_token"],
        pad_token: str = SPECIAL_TOKENS["pad_token"],
        unk_token: str = SPECIAL_TOKENS["unk_token"],
        **kwargs,
    ):
        self.sp_model = spm.SentencePieceProcessor()
        self.vocab_file = self._resolve_vocab_path(vocab_file, kwargs)
        if self.vocab_file and Path(self.vocab_file).is_file():
            self.sp_model.Load(self.vocab_file)
        if self.sp_model.get_piece_size() == 0:
            raise ValueError(f"MurzikTokenizer: missing or empty SentencePiece model ({vocab_file})")
        kwargs.setdefault("chat_template", MURZIK_CHAT_TEMPLATE)
        # Role tokens are user_defined_symbols inside SPM — do not register them as HF
        # added_tokens (that would assign ids >= vocab_size and break embedding lookup).
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs,
        )
        self._bind_special_token_ids_from_spm()

    def _spm_id(self, token: str) -> int:
        idx = self.sp_model.piece_to_id(token)
        if idx == self.sp_model.unk_id():
            raise ValueError(f"Special token {token!r} missing from SentencePiece model")
        return idx

    def _bind_special_token_ids_from_spm(self) -> None:
        self.pad_token_id = self._spm_id(self.pad_token)
        self.bos_token_id = self._spm_id(self.bos_token)
        self.eos_token_id = self._spm_id(self.eos_token)
        self.unk_token_id = self._spm_id(self.unk_token)

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def __len__(self) -> int:
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        return {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: str) -> list[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        if index < 0 or index >= self.sp_model.get_piece_size():
            raise IndexError(f"Token id {index} out of SPM range")
        return self.sp_model.id_to_piece(index)

    def encode_murzik_prompt(self, system: str, user: str, assistant: str | None = None) -> list[int]:
        """Build the exact LlamaFactory `murzik` template token ids."""
        text = f"<|murzik|><|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
        if assistant is not None:
            text += f"{assistant}<|end|>"
        return self.encode(text, add_special_tokens=False)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.sp_model.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        if token_ids_1 is not None:
            return ([0] * len(token_ids_0)) + ([1] + [0] * (len(token_ids_1) - 1))
        return [0] * len(token_ids_0)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * (len(token_ids_0) + len(token_ids_1))

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        out = Path(save_directory) / f"{filename_prefix or ''}murzik.model"
        src = Path(self.vocab_file) if self.vocab_file else out
        if not src.is_file():
            raise ValueError(f"Cannot save vocabulary, missing {src}")
        import shutil

        if src.resolve() == out.resolve():
            return (str(out),)
        shutil.copy2(src, out)
        self.vocab_file = str(out)
        return (str(out),)
