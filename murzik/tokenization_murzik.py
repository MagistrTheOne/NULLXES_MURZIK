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


class MurzikTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        bos_token: str = SPECIAL_TOKENS["bos_token"],
        eos_token: str = SPECIAL_TOKENS["eos_token"],
        pad_token: str = SPECIAL_TOKENS["pad_token"],
        unk_token: str = SPECIAL_TOKENS["unk_token"],
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        if vocab_file and Path(vocab_file).exists():
            self.sp_model.Load(vocab_file)
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        return {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: str) -> list[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.sp_model.id_to_piece(index)

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
        if self.vocab_file:
            import shutil
            shutil.copy(self.vocab_file, out)
        return (str(out),)
