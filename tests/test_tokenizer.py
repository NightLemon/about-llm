from __future__ import annotations

import pytest

from about_llm.from_scratch.tokenizer import ByteBPETokenizer, ByteTokenizer

pytestmark = pytest.mark.formula


@pytest.mark.parametrize("text", ["hello", "你好 LLM", "🙂\ncode\t123", ""])
def test_byte_tokenizer_round_trip(text: str) -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_tokenizer_rejects_invalid_ids() -> None:
    with pytest.raises(ValueError, match="invalid sample"):
        ByteTokenizer().decode([0, 256])

    with pytest.raises(ValueError, match="invalid sample"):
        ByteTokenizer().decode([True])


@pytest.mark.parametrize(
    "text",
    ["banana bandana", "你好，语言模型", "🙂\ncode\t123", "e\u0301 ≠ é", ""],  # noqa: RUF001
)
def test_byte_bpe_round_trip_preserves_utf8_text(text: str) -> None:
    tokenizer = ByteBPETokenizer.train(
        ["banana bandana", "你好你好", "code code", "🙂🙂"],
        vocab_size=280,
        min_pair_frequency=2,
    )

    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text
    assert tokenizer.decode_bytes(token_ids) == text.encode("utf-8")


def test_byte_bpe_learns_ranked_non_overlapping_merges() -> None:
    tokenizer = ByteBPETokenizer.train(
        ["abab"], vocab_size=258, min_pair_frequency=1
    )

    assert tokenizer.merges == ((ord("a"), ord("b")), (256, 256))
    assert tokenizer.encode("abab") == [257]
    assert tokenizer.token_bytes(256) == b"ab"
    assert tokenizer.token_bytes(257) == b"abab"
    assert tokenizer.decode([257]) == "abab"


def test_byte_bpe_training_is_deterministic_under_frequency_ties() -> None:
    first = ByteBPETokenizer.train(
        ["abac"], vocab_size=257, min_pair_frequency=1
    )
    second = ByteBPETokenizer.train(
        ["abac"], vocab_size=257, min_pair_frequency=1
    )

    assert first.merges == second.merges == ((ord("a"), ord("b")),)


def test_byte_bpe_does_not_count_pairs_across_document_boundaries() -> None:
    tokenizer = ByteBPETokenizer.train(
        ["a", "b"], vocab_size=257, min_pair_frequency=1
    )

    assert tokenizer.vocab_size == 256
    assert tokenizer.merges == ()


def test_byte_bpe_stops_when_no_pair_meets_minimum_frequency() -> None:
    tokenizer = ByteBPETokenizer.train(
        ["abcd"], vocab_size=300, min_pair_frequency=2
    )

    assert tokenizer.vocab_size == 256


@pytest.mark.parametrize(
    ("documents", "kwargs", "error", "message"),
    [
        ("one document", {}, TypeError, "iterable of strings"),
        ([""], {}, ValueError, "non-empty"),
        (["text"], {"vocab_size": 255}, ValueError, "vocab_size"),
        (["text"], {"min_pair_frequency": 0}, ValueError, "min_pair_frequency"),
        (["text", 1], {}, TypeError, r"documents\[1\]"),
    ],
)
def test_byte_bpe_training_rejects_ambiguous_or_invalid_inputs(
    documents: object,
    kwargs: dict[str, int],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ByteBPETokenizer.train(documents, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "merges",
    [
        ((256, 0),),
        ((0, True),),
        ((1, 2), (1, 2)),
    ],
)
def test_byte_bpe_rejects_invalid_merge_graph(
    merges: tuple[tuple[int, int], ...]
) -> None:
    with pytest.raises(ValueError):
        ByteBPETokenizer(merges)


def test_byte_bpe_rejects_unknown_or_boolean_token_ids() -> None:
    tokenizer = ByteBPETokenizer()

    with pytest.raises(ValueError, match="invalid sample"):
        tokenizer.decode([256])
    with pytest.raises(ValueError, match="invalid sample"):
        tokenizer.decode([False])
