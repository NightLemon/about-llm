"""Transparent text metrics suitable for deterministic baselines."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from about_llm.evaluation.runner import EvaluationCase

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", re.UNICODE)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return " ".join(normalized.split())


def literal_exact_match(case: EvaluationCase, output: str) -> float:
    """Compare decoded strings without Unicode, case, or whitespace normalization."""

    return float(output == case.expected)


def normalized_exact_match(case: EvaluationCase, output: str) -> float:
    return float(normalize_text(output) == normalize_text(case.expected))


def token_f1(case: EvaluationCase, output: str) -> float:
    predicted = TOKEN_PATTERN.findall(normalize_text(output))
    expected = TOKEN_PATTERN.findall(normalize_text(case.expected))
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)
