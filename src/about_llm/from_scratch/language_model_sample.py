"""Trace one string from UTF-8 bytes to a causal language-model target."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from about_llm.from_scratch.tokenizer import ByteBPETokenizer

SCHEMA_VERSION = "about-llm.language-model-sample.v1"
DEFAULT_TEXT = "你好🙂!"
DEFAULT_TRAINING_DOCUMENTS = ("你好🙂你好🙂",)


def build_language_model_sample(
    *,
    text: str = DEFAULT_TEXT,
    training_documents: Iterable[str] = DEFAULT_TRAINING_DOCUMENTS,
    vocab_size: int = 280,
    min_pair_frequency: int = 2,
) -> dict[str, Any]:
    """Build a deterministic teaching trace without running a language model."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        raise ValueError("text must not be empty")
    if isinstance(training_documents, (str, bytes)):
        raise TypeError("training_documents must be an iterable of strings")
    documents = tuple(training_documents)
    tokenizer = ByteBPETokenizer.train(
        documents,
        vocab_size=vocab_size,
        min_pair_frequency=min_pair_frequency,
    )
    text_token_ids = tokenizer.encode(text)
    if tokenizer.decode(text_token_ids) != text:
        raise AssertionError("Byte BPE round trip changed the sample text")

    bos_id = tokenizer.vocab_size
    eos_id = bos_id + 1
    pad_id = bos_id + 2
    special_token_ids = {"BOS": bos_id, "EOS": eos_id, "PAD": pad_id}
    full_sequence = [bos_id, *text_token_ids, eos_id, pad_id]
    model_input_ids = full_sequence[:-1]
    labels = full_sequence[1:]
    loss_mask = [target_id != pad_id for target_id in labels]
    causal_attention_mask = [
        [key_position <= query_position for key_position in range(len(model_input_ids))]
        for query_position in range(len(model_input_ids))
    ]

    pieces = [
        _piece_snapshot(tokenizer, position, token_id)
        for position, token_id in enumerate(text_token_ids)
    ]
    rows = [
        {
            "position": position,
            "input_id": input_id,
            "input_piece": _display_piece(
                tokenizer, input_id, special_token_ids=special_token_ids
            ),
            "target_id": target_id,
            "target_piece": _display_piece(
                tokenizer, target_id, special_token_ids=special_token_ids
            ),
            "visible_input_positions": list(range(position + 1)),
            "included_in_loss": loss_mask[position],
        }
        for position, (input_id, target_id) in enumerate(
            zip(model_input_ids, labels, strict=True)
        )
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "text": text,
        "counts": {
            "unicode_code_points": len(text),
            "utf8_bytes": len(text.encode("utf-8")),
            "model_text_tokens": len(text_token_ids),
        },
        "tokenizer": {
            "kind": "deterministic byte-level BPE teaching reference",
            "normalization": "none",
            "pre_tokenization": "none",
            "training_documents": list(documents),
            "base_vocabulary_size": tokenizer.base_vocab_size,
            "learned_merge_count": len(tokenizer.merges),
            "text_token_ids": text_token_ids,
            "pieces": pieces,
            "round_trip_matches_input": True,
        },
        "teaching_model": {
            "special_token_ids": special_token_ids,
            "embedding_row_count_required": tokenizer.vocab_size + len(special_token_ids),
            "full_sequence_ids": full_sequence,
            "model_input_ids": model_input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "effective_target_count": sum(loss_mask),
            "causal_attention_mask": causal_attention_mask,
            "position_trace": rows,
        },
        "scope": {
            "byte_bpe_trained_and_executed": True,
            "utf8_round_trip_checked": True,
            "shift_and_masks_constructed": True,
            "special_tokens_belong_to_this_walkthrough_only": True,
            "embedding_or_language_model_executed": False,
            "logits_loss_or_perplexity_computed": False,
            "checkpoint_compatibility_proved": False,
        },
    }


def _piece_snapshot(
    tokenizer: ByteBPETokenizer, position: int, token_id: int
) -> dict[str, object]:
    raw = tokenizer.token_bytes(token_id)
    return {
        "position": position,
        "token_id": token_id,
        "bytes_hex": raw.hex(),
        "utf8_preview": raw.decode("utf-8", errors="replace"),
    }


def _display_piece(
    tokenizer: ByteBPETokenizer,
    token_id: int,
    *,
    special_token_ids: Mapping[str, int],
) -> str:
    for name, special_id in special_token_ids.items():
        if token_id == special_id:
            return f"<{name}>"
    return tokenizer.token_bytes(token_id).decode("utf-8", errors="replace")
