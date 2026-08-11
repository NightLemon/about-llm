"""Metrics for machine-readable outputs and source citation syntax."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from about_llm.evaluation.runner import EvaluationCase
from about_llm.llmops import canonical_json_bytes
from about_llm.rag.citations import audit_citations


def json_schema_metric(case: EvaluationCase, output: str) -> float:
    """Return 1 for valid JSON conforming to metadata.output_schema, otherwise 0."""
    schema = case.metadata.get("output_schema")
    if not isinstance(schema, Mapping):
        raise ValueError(f"case {case.case_id!r} needs an output_schema mapping")
    plain_schema = json.loads(canonical_json_bytes(schema))
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("install the 'evaluation' extra for JSON Schema metrics") from error
    try:
        instance: Any = json.loads(output)
        validator = jsonschema.validators.validator_for(plain_schema)
        validator.check_schema(plain_schema)
        validator(plain_schema).validate(instance)
    except (json.JSONDecodeError, jsonschema.ValidationError, jsonschema.SchemaError):
        return 0.0
    return 1.0


def citation_syntax_metric(case: EvaluationCase, output: str) -> float:
    """Score known citation ids and paragraph coverage; not semantic entailment."""
    raw_ids = case.metadata.get("valid_source_ids")
    if (
        not isinstance(raw_ids, Sequence)
        or isinstance(raw_ids, str)
        or not all(isinstance(value, str) for value in raw_ids)
    ):
        raise ValueError(f"case {case.case_id!r} needs a string-list valid_source_ids")
    audit = audit_citations(output, raw_ids)
    known_count = len(audit.cited_source_ids) - len(audit.unknown_source_ids)
    precision = known_count / len(audit.cited_source_ids) if audit.cited_source_ids else 0.0
    paragraph_count = len(audit.uncited_paragraphs) + _cited_paragraph_count(output)
    coverage = 1 - len(audit.uncited_paragraphs) / paragraph_count if paragraph_count else 1.0
    return (precision + coverage) / 2


def _cited_paragraph_count(output: str) -> int:
    import re

    return sum(
        1
        for paragraph in re.split(r"\n\s*\n", output.strip())
        if re.search(r"\[[A-Z][A-Z0-9_-]*\d+\]", paragraph)
    )
