"""A deliberately small, fully reversible UTF-8 byte tokenizer."""

from __future__ import annotations

from collections.abc import Iterable


class ByteTokenizer:
    """Map UTF-8 bytes directly to ids in [0, 255].

    This tokenizer is inefficient for real language models but ideal for
    teaching: every input is representable, the vocabulary is fixed, and the
    encode/decode contract is visible without external model files.
    """

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Iterable[int], *, errors: str = "strict") -> str:
        ids = list(token_ids)
        invalid = [
            token_id for token_id in ids if not isinstance(token_id, int) or not 0 <= token_id < 256
        ]
        if invalid:
            raise ValueError(
                f"token ids must be integers in [0, 255], invalid sample: {invalid[:3]}"
            )
        return bytes(ids).decode("utf-8", errors=errors)
