"""Deterministic lexical tokenization for the dependency-free BM25 baseline."""

from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", re.UNICODE)


def lexical_tokens(text: str) -> list[str]:
    """Lowercase Latin terms and emit each CJK character as a token.

    This is intentionally a transparent baseline, not a production Chinese
    segmenter. Character tokens give non-zero recall without a dictionary and
    make its limitations easy to measure against learned retrievers.
    """
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
