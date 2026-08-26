"""Trace one string from UTF-8 bytes to causal-LM labels and masks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from about_llm.from_scratch.language_model_sample import (
    DEFAULT_TEXT,
    DEFAULT_TRAINING_DOCUMENTS,
    build_language_model_sample,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to trace")
    parser.add_argument(
        "--training-text",
        action="append",
        help="one Byte-BPE training document; repeat to provide more documents",
    )
    parser.add_argument("--vocab-size", type=int, default=280)
    parser.add_argument("--min-pair-frequency", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    training_documents = (
        tuple(args.training_text)
        if args.training_text
        else DEFAULT_TRAINING_DOCUMENTS
    )
    report = build_language_model_sample(
        text=args.text,
        training_documents=training_documents,
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
