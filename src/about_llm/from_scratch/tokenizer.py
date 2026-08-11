"""Transparent, reversible UTF-8 byte and byte-level BPE tokenizers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise


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
            token_id
            for token_id in ids
            if isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < 256
        ]
        if invalid:
            raise ValueError(
                f"token ids must be integers in [0, 255], invalid sample: {invalid[:3]}"
            )
        return bytes(ids).decode("utf-8", errors=errors)


class ByteBPETokenizer:
    """A deterministic byte-level BPE reference implementation.

    The base vocabulary is the 256 raw byte values. Every learned merge creates
    one new token whose id is ``256 + merge_rank``. Training never merges across
    document boundaries and resolves equal-frequency pairs lexicographically.

    This deliberately omits normalization, pre-tokenization, special tokens,
    offset mapping, and production indexing. It teaches the BPE mechanism; it is
    not a drop-in replacement for a checkpoint's tokenizer.
    """

    base_vocab_size = 256

    def __init__(self, merges: Iterable[tuple[int, int]] = ()) -> None:
        snapshot: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        token_bytes = [bytes([value]) for value in range(self.base_vocab_size)]
        for rank, pair in enumerate(merges):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("each BPE merge must be a (left_id, right_id) tuple")
            left, right = pair
            new_id = self.base_vocab_size + rank
            if any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < new_id
                for token_id in pair
            ):
                raise ValueError(
                    f"merge rank {rank} references an unavailable token id: {pair}"
                )
            if pair in seen:
                raise ValueError(f"duplicate BPE merge pair: {pair}")
            seen.add(pair)
            snapshot.append((left, right))
            token_bytes.append(token_bytes[left] + token_bytes[right])
        self._merges = tuple(snapshot)
        self._token_bytes = tuple(token_bytes)

    @property
    def merges(self) -> tuple[tuple[int, int], ...]:
        """Return merge rules in rank order."""

        return self._merges

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self._merges)

    @classmethod
    def train(
        cls,
        documents: Iterable[str],
        *,
        vocab_size: int = 512,
        min_pair_frequency: int = 2,
    ) -> ByteBPETokenizer:
        """Learn merge rules from independent UTF-8 documents.

        ``vocab_size`` is an upper bound: training stops early when no adjacent
        pair reaches ``min_pair_frequency``. Passing one string instead of an
        iterable of documents is rejected because it would silently turn every
        character into a separate document and change pair counts.
        """

        if isinstance(documents, (str, bytes)):
            raise TypeError("documents must be an iterable of strings, not one string")
        _validate_training_integer(
            vocab_size,
            name="vocab_size",
            minimum=cls.base_vocab_size,
        )
        _validate_training_integer(
            min_pair_frequency,
            name="min_pair_frequency",
            minimum=1,
        )
        sequences: list[list[int]] = []
        for index, document in enumerate(documents):
            if not isinstance(document, str):
                raise TypeError(
                    f"documents[{index}] must be str, got {type(document).__name__}"
                )
            sequences.append(list(document.encode("utf-8")))
        if not any(sequences):
            raise ValueError("training requires at least one non-empty UTF-8 document")

        merges: list[tuple[int, int]] = []
        while cls.base_vocab_size + len(merges) < vocab_size:
            counts = _count_adjacent_pairs(sequences)
            eligible = [
                pair for pair, count in counts.items() if count >= min_pair_frequency
            ]
            if not eligible:
                break
            pair = min(eligible, key=lambda item: (-counts[item], item))
            new_id = cls.base_vocab_size + len(merges)
            sequences = [
                _merge_non_overlapping(sequence, pair, new_id)
                for sequence in sequences
            ]
            merges.append(pair)
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        """Encode text by applying learned merge rules in rank order."""

        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        token_ids = list(text.encode("utf-8"))
        for rank, pair in enumerate(self._merges):
            token_ids = _merge_non_overlapping(
                token_ids,
                pair,
                self.base_vocab_size + rank,
            )
        return token_ids

    def decode_bytes(self, token_ids: Iterable[int]) -> bytes:
        """Expand token ids back to raw bytes without assuming valid UTF-8."""

        ids = list(token_ids)
        invalid = [
            token_id
            for token_id in ids
            if isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < self.vocab_size
        ]
        if invalid:
            raise ValueError(
                f"token ids must be integers in [0, {self.vocab_size - 1}], "
                f"invalid sample: {invalid[:3]}"
            )
        return b"".join(self._token_bytes[token_id] for token_id in ids)

    def decode(self, token_ids: Iterable[int], *, errors: str = "strict") -> str:
        """Expand ids and decode the resulting UTF-8 byte stream."""

        return self.decode_bytes(token_ids).decode("utf-8", errors=errors)

    def token_bytes(self, token_id: int) -> bytes:
        """Inspect the byte expansion of one vocabulary entry."""

        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < self.vocab_size
        ):
            raise ValueError(f"token_id must be an integer in [0, {self.vocab_size - 1}]")
        return self._token_bytes[token_id]


def _validate_training_integer(value: int, *, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _count_adjacent_pairs(sequences: Sequence[Sequence[int]]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for sequence in sequences:
        counts.update(pairwise(sequence))
    return counts


def _merge_non_overlapping(
    sequence: Sequence[int], pair: tuple[int, int], new_id: int
) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == pair[0]
            and sequence[index + 1] == pair[1]
        ):
            merged.append(new_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged
