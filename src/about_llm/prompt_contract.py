"""Small, explicit output contract for the prompting chapter's extraction example."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, NoReturn, cast

_TOP_LEVEL_FIELDS = {"status", "party", "signed_on", "currency", "evidence"}
_VALUE_FIELDS = ("party", "signed_on", "currency")
_EVIDENCE_FIELDS = {"field", "quote", "start_char", "end_char"}
_STATUSES = {"complete", "insufficient_evidence", "conflict"}
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CURRENCY_CODE = re.compile(r"(?<![A-Z])[A-Z]{3}(?![A-Z])")
_ZH_DATE = re.compile(
    r"^\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*$"
)


@dataclass(frozen=True)
class ContractValidationReport:
    """Validation result with each evidence layer kept separate."""

    strict_json_valid: bool
    closed_shape_valid: bool
    exact_spans_valid: bool
    field_semantics_valid: bool
    decision: str
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "strict_json_valid": self.strict_json_valid,
            "closed_shape_valid": self.closed_shape_valid,
            "exact_spans_valid": self.exact_spans_valid,
            "field_semantics_valid": self.field_semantics_valid,
            "decision": self.decision,
            "errors": list(self.errors),
        }


def validate_contract_extraction(
    source_text: str,
    raw_output: str,
) -> ContractValidationReport:
    """Validate the fixed teaching contract without calling a model.

    This intentionally checks a narrow output protocol. It is not a general
    legal-document validator or a JSON Schema implementation.
    """

    if not isinstance(source_text, str) or not source_text:
        raise ValueError("source_text must be a non-empty string")
    if not isinstance(raw_output, str) or not raw_output:
        raise ValueError("raw_output must be a non-empty string")

    try:
        output = _strict_json_object(raw_output)
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        return ContractValidationReport(
            strict_json_valid=False,
            closed_shape_valid=False,
            exact_spans_valid=False,
            field_semantics_valid=False,
            decision="reject",
            errors=(f"json:{error}",),
        )

    shape_errors = _closed_shape_errors(output)
    if shape_errors:
        return ContractValidationReport(
            strict_json_valid=True,
            closed_shape_valid=False,
            exact_spans_valid=False,
            field_semantics_valid=False,
            decision="repair_or_reject",
            errors=tuple(shape_errors),
        )

    evidence = cast(list[dict[str, object]], output["evidence"])
    span_errors = _span_errors(source_text, evidence)
    semantic_errors = _semantic_errors(output, evidence)
    errors = (*span_errors, *semantic_errors)
    exact_spans_valid = not span_errors
    semantics_valid = exact_spans_valid and not semantic_errors
    return ContractValidationReport(
        strict_json_valid=True,
        closed_shape_valid=True,
        exact_spans_valid=exact_spans_valid,
        field_semantics_valid=semantics_valid,
        decision="accept" if semantics_valid else "reject",
        errors=errors,
    )


def _strict_json_object(raw_output: str) -> dict[str, object]:
    value: Any = json.loads(
        raw_output,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return cast(dict[str, object], value)


def _closed_shape_errors(output: dict[str, object]) -> list[str]:
    errors: list[str] = []
    keys = set(output)
    if keys != _TOP_LEVEL_FIELDS:
        missing = sorted(_TOP_LEVEL_FIELDS - keys)
        unknown = sorted(keys - _TOP_LEVEL_FIELDS)
        errors.append(f"shape:top_level_fields:missing={missing}:unknown={unknown}")
        return errors

    status = output["status"]
    if not isinstance(status, str) or status not in _STATUSES:
        errors.append("shape:status")
    for field in _VALUE_FIELDS:
        value = output[field]
        if value is not None and not isinstance(value, str):
            errors.append(f"shape:{field}")
    signed_on = output["signed_on"]
    if isinstance(signed_on, str) and not _is_iso_date(signed_on):
        errors.append("shape:signed_on_format")
    currency = output["currency"]
    if isinstance(currency, str) and not re.fullmatch(r"[A-Z]{3}", currency):
        errors.append("shape:currency_format")

    raw_evidence = output["evidence"]
    if not isinstance(raw_evidence, list):
        errors.append("shape:evidence")
        return errors
    for index, item in enumerate(raw_evidence):
        if not isinstance(item, dict) or set(item) != _EVIDENCE_FIELDS:
            errors.append(f"shape:evidence[{index}]:fields")
            continue
        field = item["field"]
        quote = item["quote"]
        start = item["start_char"]
        end = item["end_char"]
        if not isinstance(field, str) or field not in _VALUE_FIELDS:
            errors.append(f"shape:evidence[{index}]:field")
        if not isinstance(quote, str) or not quote:
            errors.append(f"shape:evidence[{index}]:quote")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            errors.append(f"shape:evidence[{index}]:span")
    return errors


def _span_errors(
    source_text: str,
    evidence: list[dict[str, object]],
) -> tuple[str, ...]:
    errors: list[str] = []
    for index, item in enumerate(evidence):
        start = cast(int, item["start_char"])
        end = cast(int, item["end_char"])
        quote = cast(str, item["quote"])
        if end > len(source_text) or source_text[start:end] != quote:
            errors.append(f"span:evidence[{index}]")
    return tuple(errors)


def _semantic_errors(
    output: dict[str, object],
    evidence: list[dict[str, object]],
) -> tuple[str, ...]:
    by_field: dict[str, list[dict[str, object]]] = {
        field: [] for field in _VALUE_FIELDS
    }
    for item in evidence:
        by_field[cast(str, item["field"])].append(item)

    errors: list[str] = []
    status = cast(str, output["status"])
    missing_fields = [field for field in _VALUE_FIELDS if output[field] is None]

    if status == "complete" and missing_fields:
        errors.append("semantic:status_complete_has_missing_fields")
    if status == "insufficient_evidence" and not missing_fields:
        errors.append("semantic:status_insufficient_without_missing_field")

    conflict_fields = [
        field
        for field in missing_fields
        if _has_distinct_conflict_candidates(field, by_field[field])
    ]
    if status == "conflict" and not conflict_fields:
        errors.append("semantic:conflict_requires_distinct_candidates")
    if status != "conflict" and conflict_fields:
        errors.append("semantic:multiple_candidates_require_conflict")

    for field in _VALUE_FIELDS:
        value = output[field]
        items = by_field[field]
        if value is None:
            if items and field not in conflict_fields:
                errors.append(f"semantic:null_field_has_evidence:{field}")
            continue
        if len(items) != 1:
            errors.append(f"semantic:evidence_count:{field}")
            continue
        quote = cast(str, items[0]["quote"])
        if not _quote_supports(field, cast(str, value), quote):
            errors.append(f"semantic:unsupported_value:{field}")
    return tuple(errors)


def _quote_supports(field: str, value: str, quote: str) -> bool:
    if field == "party":
        return value in quote
    if field == "signed_on":
        return _normalize_date_quote(quote) == value
    if field == "currency":
        return re.search(
            rf"(?<![A-Z]){re.escape(value)}(?![A-Z])",
            quote.upper(),
        ) is not None
    raise AssertionError(f"unexpected extraction field {field!r}")


def _has_distinct_conflict_candidates(
    field: str,
    items: list[dict[str, object]],
) -> bool:
    positions = {
        (cast(int, item["start_char"]), cast(int, item["end_char"]))
        for item in items
    }
    candidates = [_candidate_value(field, cast(str, item["quote"])) for item in items]
    return (
        len(positions) >= 2
        and all(candidate is not None for candidate in candidates)
        and len(set(candidates)) >= 2
    )


def _candidate_value(field: str, quote: str) -> str | None:
    if field == "party":
        return quote.strip() or None
    if field == "signed_on":
        return _normalize_date_quote(quote)
    if field == "currency":
        matches = set(_CURRENCY_CODE.findall(quote.upper()))
        return next(iter(matches)) if len(matches) == 1 else None
    raise AssertionError(f"unexpected extraction field {field!r}")


def _normalize_date_quote(quote: str) -> str | None:
    if _is_iso_date(quote):
        return quote
    matched = _ZH_DATE.fullmatch(quote)
    if matched is None:
        return None
    year, month, day = (int(part) for part in matched.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _is_iso_date(value: str) -> bool:
    if not _ISO_DATE.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate object key {key!r}")
        output[key] = value
    return output


def _reject_nonfinite_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r}")
