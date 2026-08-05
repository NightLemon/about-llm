from __future__ import annotations

import pytest

from about_llm.from_scratch.tokenizer import ByteTokenizer


@pytest.mark.parametrize("text", ["hello", "你好 LLM", "🙂\ncode\t123", ""])
def test_byte_tokenizer_round_trip(text: str) -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_tokenizer_rejects_invalid_ids() -> None:
    with pytest.raises(ValueError, match="invalid sample"):
        ByteTokenizer().decode([0, 256])
