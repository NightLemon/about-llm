"""Train and inspect the repository's deterministic byte-level BPE reference."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from about_llm.from_scratch import ByteBPETokenizer

DEFAULT_DOCUMENTS = (
    "low lower lowest",
    "newer wider lower",
    "你好你好，语言模型",  # noqa: RUF001
    "token tokenization tokenizer",
)
EVIDENCE_BOUNDARY = (
    "This CPU demo verifies deterministic merge learning and UTF-8 round trips for "
    "the supplied strings. It has no normalization, pre-tokenizer, special tokens, "
    "offset map, checkpoint compatibility, throughput claim, or target-corpus evidence."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        action="append",
        help="one independent training document; repeat to supply a corpus",
    )
    parser.add_argument(
        "--sample",
        action="append",
        help="text to encode after training; defaults to the training documents",
    )
    parser.add_argument("--vocab-size", type=int, default=280)
    parser.add_argument("--min-pair-frequency", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    documents = tuple(args.text) if args.text else DEFAULT_DOCUMENTS
    samples = tuple(args.sample) if args.sample else documents
    tokenizer = ByteBPETokenizer.train(
        documents,
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
    )
    merges = [
        {
            "rank": rank,
            "left_id": left,
            "right_id": right,
            "new_id": tokenizer.base_vocab_size + rank,
            "bytes_hex": tokenizer.token_bytes(tokenizer.base_vocab_size + rank).hex(),
            "utf8_preview": tokenizer.token_bytes(
                tokenizer.base_vocab_size + rank
            ).decode("utf-8", errors="replace"),
        }
        for rank, (left, right) in enumerate(tokenizer.merges)
    ]
    encoded_samples = []
    for text in samples:
        token_ids = tokenizer.encode(text)
        encoded_samples.append(
            {
                "text": text,
                "utf8_bytes": len(text.encode("utf-8")),
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "round_trip": tokenizer.decode(token_ids) == text,
            }
        )
    print(
        json.dumps(
            {
                "implementation": "about-llm.byte-bpe-reference.v1",
                "base_vocabulary": "raw byte ids 0..255",
                "normalization": "none",
                "pre_tokenization": "none",
                "requested_vocab_size": args.vocab_size,
                "actual_vocab_size": tokenizer.vocab_size,
                "min_pair_frequency": args.min_pair_frequency,
                "document_count": len(documents),
                "merges": merges,
                "samples": encoded_samples,
                "evidence_boundary": EVIDENCE_BOUNDARY,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
