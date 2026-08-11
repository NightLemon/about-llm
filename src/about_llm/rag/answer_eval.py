"""Recorded RAG answer artifacts with explicit semantic-judgment boundaries."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn, cast

from about_llm.llmops import artifact_fingerprint


class AnswerAction(str, Enum):
    """Observable terminal action of one RAG answer attempt."""

    ANSWER = "answer"
    ABSTAIN = "abstain"
    ERROR = "error"


class ClaimVerdict(str, Enum):
    """Externally supplied claim/evidence judgment; never inferred here."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"
    UNJUDGED = "unjudged"


@dataclass(frozen=True)
class RecordedClaim:
    """One atomic claim, its cited stable source ids, and a supplied verdict."""

    claim_id: str
    text: str
    source_ids: tuple[str, ...]
    verdict: ClaimVerdict
    judgment_source: str | None = None

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise ValueError("claim_id cannot be empty")
        if not self.text.strip():
            raise ValueError(f"claim {self.claim_id!r} text cannot be empty")
        _validate_unique_nonempty(self.source_ids, f"claim {self.claim_id!r} source_ids")
        if self.verdict is ClaimVerdict.UNJUDGED:
            if self.judgment_source is not None:
                raise ValueError("an unjudged claim cannot name a judgment_source")
        elif self.judgment_source is None or not self.judgment_source.strip():
            raise ValueError("a judged claim needs a non-empty judgment_source")

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "source_ids": list(self.source_ids),
            "verdict": self.verdict.value,
            "judgment_source": self.judgment_source,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, location: str) -> RecordedClaim:
        _validate_fields(
            value,
            required={"claim_id", "text", "source_ids", "verdict"},
            optional={"judgment_source"},
            location=location,
        )
        claim_id = _required_string(value, "claim_id", location)
        try:
            verdict = ClaimVerdict(_required_string(value, "verdict", location))
        except ValueError as error:
            allowed = [item.value for item in ClaimVerdict]
            raise ValueError(f"{location}: verdict must be one of {allowed}") from error
        judgment_source = value.get("judgment_source")
        if judgment_source is not None and not isinstance(judgment_source, str):
            raise ValueError(f"{location}: judgment_source must be a string or null")
        return cls(
            claim_id=claim_id,
            text=_required_string(value, "text", location),
            source_ids=_string_array(value.get("source_ids"), "source_ids", location),
            verdict=verdict,
            judgment_source=judgment_source,
        )


@dataclass(frozen=True)
class RecordedAnswer:
    """One replayable RAG output; this stores evidence, not model execution."""

    query_id: str
    action: AnswerAction
    context_source_ids: tuple[str, ...]
    claims: tuple[RecordedClaim, ...]
    missing_information: tuple[str, ...]
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id cannot be empty")
        _validate_unique_nonempty(self.context_source_ids, "context_source_ids")
        _validate_unique_nonempty(self.missing_information, "missing_information")
        claim_ids = tuple(claim.claim_id for claim in self.claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(f"query {self.query_id!r} contains duplicate claim_id values")
        if self.action is AnswerAction.ANSWER:
            if not self.claims:
                raise ValueError("answer action requires at least one atomic claim")
            if self.error_type is not None:
                raise ValueError("answer action cannot contain error_type")
        elif self.action is AnswerAction.ABSTAIN:
            if self.claims:
                raise ValueError("abstain action cannot contain factual claims")
            if not self.missing_information:
                raise ValueError("abstain action needs explicit missing_information")
            if self.error_type is not None:
                raise ValueError("abstain action cannot contain error_type")
        else:
            if self.claims:
                raise ValueError("error action cannot contain factual claims")
            if self.error_type is None or not self.error_type.strip():
                raise ValueError("error action needs a non-empty error_type")

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "action": self.action.value,
            "context_source_ids": list(self.context_source_ids),
            "claims": [claim.to_dict() for claim in self.claims],
            "missing_information": list(self.missing_information),
            "error_type": self.error_type,
        }

    @property
    def record_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, location: str) -> RecordedAnswer:
        _validate_fields(
            value,
            required={
                "query_id",
                "action",
                "context_source_ids",
                "claims",
                "missing_information",
            },
            optional={"error_type"},
            location=location,
        )
        try:
            action = AnswerAction(_required_string(value, "action", location))
        except ValueError as error:
            allowed = [item.value for item in AnswerAction]
            raise ValueError(f"{location}: action must be one of {allowed}") from error
        raw_claims = value.get("claims")
        if not isinstance(raw_claims, list):
            raise ValueError(f"{location}: claims must be an array")
        claims: list[RecordedClaim] = []
        for index, raw_claim in enumerate(raw_claims):
            claim_location = f"{location}:claims[{index}]"
            if not isinstance(raw_claim, dict) or not all(
                isinstance(key, str) for key in raw_claim
            ):
                raise ValueError(f"{claim_location}: claim must be an object")
            claims.append(
                RecordedClaim.from_mapping(
                    cast(dict[str, Any], raw_claim), location=claim_location
                )
            )
        error_type = value.get("error_type")
        if error_type is not None and not isinstance(error_type, str):
            raise ValueError(f"{location}: error_type must be a string or null")
        return cls(
            query_id=_required_string(value, "query_id", location),
            action=action,
            context_source_ids=_string_array(
                value.get("context_source_ids"), "context_source_ids", location
            ),
            claims=tuple(claims),
            missing_information=_string_array(
                value.get("missing_information"), "missing_information", location
            ),
            error_type=error_type,
        )


def load_recorded_answers(path: Path) -> tuple[RecordedAnswer, ...]:
    """Load strict UTF-8 JSONL answer artifacts and reject duplicate query ids."""
    answers: list[RecordedAnswer] = []
    query_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        location = f"{path}:{line_number}"
        try:
            raw_value: Any = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{location}: invalid strict JSON: {error}") from error
        if not isinstance(raw_value, dict) or not all(
            isinstance(key, str) for key in raw_value
        ):
            raise ValueError(f"{location}: answer must be an object")
        answer = RecordedAnswer.from_mapping(
            cast(dict[str, Any], raw_value), location=location
        )
        if answer.query_id in query_ids:
            raise ValueError(f"{location}: duplicate query_id {answer.query_id!r}")
        query_ids.add(answer.query_id)
        answers.append(answer)
    if not answers:
        raise ValueError(f"{path}: answers must contain at least one non-empty JSON line")
    return tuple(answers)


def evaluate_recorded_answers(
    *,
    expected_answerable: Mapping[str, bool],
    answers: Sequence[RecordedAnswer],
    context_status: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Score recorded actions/citations plus externally supplied claim verdicts.

    This function never performs semantic entailment. A ``supported`` verdict
    is evidence supplied by the artifact and remains only as trustworthy as its
    named judgment process.
    """
    if not expected_answerable:
        raise ValueError("at least one expected query is required")
    answer_by_id = {answer.query_id: answer for answer in answers}
    if len(answer_by_id) != len(answers):
        raise ValueError("answers contain duplicate query ids")
    _validate_exact_query_join(expected_answerable, answer_by_id, "answers")
    _validate_exact_query_join(expected_answerable, context_status, "context_status")

    rows: list[dict[str, Any]] = []
    action_correct_count = 0
    answerable_action_correct_count = 0
    no_answer_action_correct_count = 0
    answerable_count = sum(expected_answerable.values())
    no_answer_count = len(expected_answerable) - answerable_count
    answered_count = 0
    error_count = 0
    grounded_answer_pass_count = 0
    recorded_gate_pass_count = 0
    claim_count = 0
    cited_claim_count = 0
    citation_count = 0
    valid_citation_count = 0
    judged_claim_count = 0
    supported_claim_count = 0
    contradicted_claim_count = 0
    insufficient_claim_count = 0
    unjudged_claim_count = 0

    for query_id in expected_answerable:
        answer = answer_by_id[query_id]
        expected_action = (
            AnswerAction.ANSWER
            if expected_answerable[query_id]
            else AnswerAction.ABSTAIN
        )
        statuses = dict(context_status[query_id])
        if set(statuses) != set(answer.context_source_ids):
            raise ValueError(
                f"context_status for query {query_id!r} must match context_source_ids"
            )
        invalid_statuses = set(statuses.values()) - {
            "visible",
            "acl_blocked",
            "missing_from_tenant_corpus",
        }
        if invalid_statuses:
            raise ValueError(
                f"context_status for query {query_id!r} has unknown values: "
                f"{sorted(invalid_statuses)}"
            )
        action_correct = answer.action is expected_action
        action_correct_count += int(action_correct)
        if expected_answerable[query_id]:
            answerable_action_correct_count += int(action_correct)
        else:
            no_answer_action_correct_count += int(action_correct)

        unauthorized_context_ids = sorted(
            source_id for source_id, status in statuses.items() if status != "visible"
        )
        visible_context = not unauthorized_context_ids
        diagnostics: list[str] = []
        if not action_correct:
            diagnostics.append("action_mismatch")
        if unauthorized_context_ids:
            diagnostics.append("context_not_authorized_or_missing")
        if answer.action is AnswerAction.ERROR:
            error_count += 1
            diagnostics.append(f"system_error:{answer.error_type}")

        claim_rows: list[dict[str, Any]] = []
        all_claims_supported = bool(answer.claims)
        all_claims_cited = bool(answer.claims)
        all_citations_valid = bool(answer.claims)
        for claim in answer.claims:
            claim_count += 1
            has_citation = bool(claim.source_ids)
            cited_claim_count += int(has_citation)
            all_claims_cited = all_claims_cited and has_citation
            invalid_source_ids = sorted(
                source_id
                for source_id in claim.source_ids
                if source_id not in statuses or statuses.get(source_id) != "visible"
            )
            citation_count += len(claim.source_ids)
            valid_citation_count += len(claim.source_ids) - len(invalid_source_ids)
            all_citations_valid = all_citations_valid and not invalid_source_ids
            is_judged = claim.verdict is not ClaimVerdict.UNJUDGED
            judged_claim_count += int(is_judged)
            supported_claim_count += int(claim.verdict is ClaimVerdict.SUPPORTED)
            contradicted_claim_count += int(claim.verdict is ClaimVerdict.CONTRADICTED)
            insufficient_claim_count += int(claim.verdict is ClaimVerdict.INSUFFICIENT)
            unjudged_claim_count += int(claim.verdict is ClaimVerdict.UNJUDGED)
            all_claims_supported = (
                all_claims_supported and claim.verdict is ClaimVerdict.SUPPORTED
            )
            if not has_citation:
                diagnostics.append(f"uncited_claim:{claim.claim_id}")
            if invalid_source_ids:
                diagnostics.append(f"invalid_claim_citation:{claim.claim_id}")
            if claim.verdict is not ClaimVerdict.SUPPORTED:
                diagnostics.append(f"claim_{claim.verdict.value}:{claim.claim_id}")
            claim_rows.append(
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "source_ids": list(claim.source_ids),
                    "invalid_source_ids": invalid_source_ids,
                    "verdict": claim.verdict.value,
                    "judgment_source": claim.judgment_source,
                }
            )

        grounding_pass = (
            answer.action is AnswerAction.ANSWER
            and visible_context
            and all_claims_cited
            and all_citations_valid
            and all_claims_supported
        )
        if answer.action is AnswerAction.ANSWER:
            answered_count += 1
            grounded_answer_pass_count += int(grounding_pass)
        recorded_gate_pass = (
            grounding_pass
            if expected_answerable[query_id]
            else answer.action is AnswerAction.ABSTAIN and visible_context
        )
        recorded_gate_pass_count += int(recorded_gate_pass)
        rows.append(
            {
                "query_id": query_id,
                "expected_action": expected_action.value,
                "actual_action": answer.action.value,
                "error_type": answer.error_type,
                "action_correct": action_correct,
                "context_source_status": dict(sorted(statuses.items())),
                "unauthorized_or_missing_context_source_ids": unauthorized_context_ids,
                "claims": claim_rows,
                "missing_information": list(answer.missing_information),
                "grounded_answer_pass": (
                    grounding_pass if answer.action is AnswerAction.ANSWER else None
                ),
                "recorded_gate_pass": recorded_gate_pass,
                "diagnostics": diagnostics,
            }
        )

    case_count = len(expected_answerable)
    return {
        "case_count": case_count,
        "answerable_case_count": answerable_count,
        "no_answer_case_count": no_answer_count,
        "answered_case_count": answered_count,
        "error_case_count": error_count,
        "coverage": answered_count / case_count,
        "action_accuracy": action_correct_count / case_count,
        "answerable_action_accuracy": _optional_ratio(
            answerable_action_correct_count, answerable_count
        ),
        "no_answer_abstention_action_accuracy": _optional_ratio(
            no_answer_action_correct_count, no_answer_count
        ),
        "grounded_answer_pass_rate": _optional_ratio(
            grounded_answer_pass_count, answered_count
        ),
        "recorded_gate_pass_rate": recorded_gate_pass_count / case_count,
        "claim_count": claim_count,
        "citation_coverage": _optional_ratio(cited_claim_count, claim_count),
        "citation_validity": _optional_ratio(valid_citation_count, citation_count),
        "claim_judgment_coverage": _optional_ratio(judged_claim_count, claim_count),
        "supported_claim_rate": _optional_ratio(
            supported_claim_count, judged_claim_count
        ),
        "claim_verdict_counts": {
            "supported": supported_claim_count,
            "contradicted": contradicted_claim_count,
            "insufficient": insufficient_claim_count,
            "unjudged": unjudged_claim_count,
        },
        "scope_warning": (
            "claim verdicts are supplied labels, not entailment inferred by this evaluator; "
            "the gate does not prove judgment reliability, answer completeness, abstention-"
            "rationale correctness, source quality, or production safety"
        ),
        "cases": rows,
    }


def _validate_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    location: str,
) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing:
        raise ValueError(f"{location}: missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{location}: unknown fields: {sorted(extra)}")


def _required_string(value: Mapping[str, Any], key: str, location: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{location}: {key} must be a non-empty string")
    return item


def _string_array(value: Any, key: str, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{location}: {key} must be an array of non-empty strings")
    strings = cast(list[str], value)
    if len(strings) != len(set(strings)):
        raise ValueError(f"{location}: {key} contains duplicate values")
    return tuple(strings)


def _validate_unique_nonempty(values: Sequence[str], label: str) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} cannot contain an empty value")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate values")


def _validate_exact_query_join(
    expected: Mapping[str, object], actual: Mapping[str, object], label: str
) -> None:
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    if missing or extra:
        raise ValueError(
            f"{label} query join mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
