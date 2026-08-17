"""Metrics for machine-readable outputs and source citation syntax."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from about_llm.evaluation.runner import EvaluationCase
from about_llm.llmops import canonical_json_bytes
from about_llm.rag.citations import audit_citations


def json_schema_metric(case: EvaluationCase, output: str) -> float:
    """Return 1 for strict JSON conforming to a valid local-reference schema."""

    schema = case.metadata.get("output_schema")
    if not isinstance(schema, Mapping):
        raise ValueError(f"case {case.case_id!r} needs an output_schema mapping")
    plain_schema = json.loads(canonical_json_bytes(schema))
    _reject_nonlocal_schema_references(plain_schema)
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("install the 'evaluation' extra for JSON Schema metrics") from error
    try:
        validator = jsonschema.validators.validator_for(plain_schema)
        validator.check_schema(plain_schema)
    except jsonschema.SchemaError as error:
        raise ValueError(f"case {case.case_id!r} has an invalid output_schema") from error
    try:
        instance = _strict_json_loads(output)
    except (json.JSONDecodeError, ValueError):
        return 0.0
    try:
        validator(plain_schema).validate(instance)
    except jsonschema.ValidationError:
        return 0.0
    return 1.0


def json_value_exact_metric(case: EvaluationCase, output: str) -> float:
    """Compare strict parsed JSON under the v1 canonical value policy.

    Object key order and insignificant JSON whitespace are ignored. Array order,
    string contents, scalar types, and the parser's integer/float distinction are
    preserved. This is expected-value equality, not business-semantic validation.
    """

    try:
        expected = _strict_json_loads(case.expected)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"case {case.case_id!r} expected must be strict JSON"
        ) from error
    try:
        actual = _strict_json_loads(output)
    except (json.JSONDecodeError, ValueError):
        return 0.0
    return float(canonical_json_bytes(actual) == canonical_json_bytes(expected))


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


def citation_evidence_span_metric(case: EvaluationCase, output: str) -> float:
    """Verify strict claim-to-source span bindings without inferring entailment.

    ``case.metadata["citation_sources"]`` must map authorized source ids to the
    exact decoded source strings used for evaluation. The output contract is a
    strict JSON object of the form::

        {"claims": [{"claim_id": "c1", "text": "...", "evidence": [
            {"source_id": "S1", "start_char": 0, "end_char": 4,
             "quote": "text"}
        ]}]}

    Offsets are zero-based, end-exclusive Python string indices. A score of one
    proves only that every non-empty claim names at least one authorized source
    span and that each quote is the exact substring at its recorded offsets. It
    does not prove that the quote supports the claim, that the claim is complete,
    or that the source is true or current.
    """

    raw_sources = case.metadata.get("citation_sources")
    if not isinstance(raw_sources, Mapping) or not raw_sources:
        raise ValueError(
            f"case {case.case_id!r} needs a non-empty citation_sources mapping"
        )
    sources: dict[str, str] = {}
    for source_id, source_text in raw_sources.items():
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(
                f"case {case.case_id!r} citation_sources needs non-empty string ids"
            )
        if not isinstance(source_text, str) or not source_text:
            raise ValueError(
                f"case {case.case_id!r} citation source {source_id!r} "
                "needs non-empty string text"
            )
        sources[source_id] = source_text

    try:
        value = _strict_json_loads(output)
    except (json.JSONDecodeError, ValueError):
        return 0.0
    if not isinstance(value, Mapping) or set(value) != {"claims"}:
        return 0.0
    claims = value.get("claims")
    if not isinstance(claims, list) or not claims:
        return 0.0

    seen_claim_ids: set[str] = set()
    for claim in claims:
        if not isinstance(claim, Mapping) or set(claim) != {
            "claim_id",
            "text",
            "evidence",
        }:
            return 0.0
        claim_id = claim.get("claim_id")
        claim_text = claim.get("text")
        evidence = claim.get("evidence")
        if (
            not isinstance(claim_id, str)
            or not claim_id.strip()
            or claim_id in seen_claim_ids
            or not isinstance(claim_text, str)
            or not claim_text.strip()
            or not isinstance(evidence, list)
            or not evidence
        ):
            return 0.0
        seen_claim_ids.add(claim_id)

        seen_spans: set[tuple[str, int, int, str]] = set()
        for span in evidence:
            if not isinstance(span, Mapping) or set(span) != {
                "source_id",
                "start_char",
                "end_char",
                "quote",
            }:
                return 0.0
            source_id = span.get("source_id")
            start = span.get("start_char")
            end = span.get("end_char")
            quote = span.get("quote")
            if (
                not isinstance(source_id, str)
                or source_id not in sources
                or isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or not isinstance(quote, str)
                or not quote
            ):
                return 0.0
            source_text = sources[source_id]
            if not 0 <= start < end <= len(source_text):
                return 0.0
            if source_text[start:end] != quote:
                return 0.0
            span_identity = (source_id, start, end, quote)
            if span_identity in seen_spans:
                return 0.0
            seen_spans.add(span_identity)
    return 1.0


def _cited_paragraph_count(output: str) -> int:
    import re

    return sum(
        1
        for paragraph in re.split(r"\n\s*\n", output.strip())
        if re.search(r"\[[A-Z][A-Z0-9_-]*\d+\]", paragraph)
    )


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_nonlocal_schema_references(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key == "$id":
                raise ValueError(
                    f"output_schema {child_path} is not allowed in the local-only metric"
                )
            if key in {"$ref", "$dynamicRef"} and (
                not isinstance(item, str) or not item.startswith("#")
            ):
                raise ValueError(
                    f"output_schema {child_path} must be a local fragment reference"
                )
            _reject_nonlocal_schema_references(item, child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_nonlocal_schema_references(item, f"{path}[{index}]")
