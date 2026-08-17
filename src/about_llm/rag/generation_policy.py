"""Fail-closed RAG generation publication policy and deterministic replay evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn, cast

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes
from about_llm.rag.citations import CitationContext, audit_citations, build_citation_context
from about_llm.rag.models import SearchResult
from about_llm.rag.transformers_control import (
    RAG_TRANSFORMERS_REPORT_VERSION,
    RAGTransformersControlSpec,
)

RAG_PUBLICATION_POLICY_VERSION = "about-llm.rag-publication-policy.v1"
RAG_PUBLICATION_REPLAY_REPORT_VERSION = (
    "about-llm.rag-publication-policy-replay-report.v1"
)
RAG_PUBLICATION_REPLAY_EVIDENCE_BOUNDARY = (
    "This report deterministically replays a fail-closed publication policy over an "
    "already recorded and separately verified RAG model attempt. It shows that the "
    "policy would skip generation when authorized evidence is empty and would reject "
    "a generated answer that fails the local citation-syntax gate, while preserving "
    "the source report's raw model output. It is counterfactual policy replay, not an "
    "observation of the policy wrapping the recorded Qwen runtime. Citation syntax "
    "does not prove claim-evidence entailment, and the replay does not authenticate "
    "unsigned artifacts, establish general quality, or prove production integration."
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 1_000_000
_MAX_RESPONSE_CHARACTERS = 4096


class PublicationStage(str, Enum):
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"


class PublicationAction(str, Enum):
    PUBLISH = "publish"
    ABSTAIN = "abstain"
    REJECT = "reject"


@dataclass(frozen=True)
class RAGPublicationPolicy:
    revision: str = RAG_PUBLICATION_POLICY_VERSION
    no_evidence_response: str = "无法根据提供的证据回答。"
    rejected_response: str = "无法生成满足引用要求的可验证答案。"

    def __post_init__(self) -> None:
        if self.revision != RAG_PUBLICATION_POLICY_VERSION:
            raise ValueError("publication policy revision is unsupported")
        for name, value in (
            ("no_evidence_response", self.no_evidence_response),
            ("rejected_response", self.rejected_response),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            if len(value) > _MAX_RESPONSE_CHARACTERS:
                raise ValueError(f"{name} exceeds the character limit")
        if self.no_evidence_response == self.rejected_response:
            raise ValueError("no-evidence and rejected responses must remain distinct")

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "revision": self.revision,
            "no_evidence_response": self.no_evidence_response,
            "rejected_response": self.rejected_response,
        }


@dataclass(frozen=True)
class RAGPublicationDecision:
    policy_revision: str
    stage: PublicationStage
    action: PublicationAction
    reason_code: str
    model_call_allowed: bool
    generated_output_observed: bool
    raw_output: str | None
    response_text: str
    valid_source_ids: tuple[str, ...]
    cited_source_ids: tuple[str, ...]
    unknown_source_ids: tuple[str, ...]
    uncited_paragraphs: tuple[str, ...]
    citation_syntax_passed: bool
    semantic_entailment_verified: bool = False

    def __post_init__(self) -> None:
        if self.policy_revision != RAG_PUBLICATION_POLICY_VERSION:
            raise ValueError("decision policy revision is unsupported")
        if not self.reason_code:
            raise ValueError("decision reason_code cannot be empty")
        if not isinstance(self.response_text, str) or not self.response_text.strip():
            raise ValueError("decision response_text cannot be empty")
        if len(self.response_text) > _MAX_RESPONSE_CHARACTERS:
            raise ValueError("decision response_text exceeds the character limit")
        for name, values in (
            ("valid_source_ids", self.valid_source_ids),
            ("cited_source_ids", self.cited_source_ids),
            ("unknown_source_ids", self.unknown_source_ids),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} contains an invalid source id")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} contains duplicate source ids")
        if self.semantic_entailment_verified:
            raise ValueError("publication policy v1 cannot verify semantic entailment")

        if self.stage is PublicationStage.PRE_GENERATION:
            if (
                self.action is not PublicationAction.ABSTAIN
                or self.reason_code != "no_authorized_evidence"
                or self.model_call_allowed
                or self.generated_output_observed
                or self.raw_output is not None
                or self.valid_source_ids
                or self.cited_source_ids
                or self.unknown_source_ids
                or self.uncited_paragraphs
                or self.citation_syntax_passed
            ):
                raise ValueError("pre-generation abstention decision is inconsistent")
            return

        if not self.valid_source_ids:
            raise ValueError("post-generation decision requires authorized source ids")
        if not self.model_call_allowed or not self.generated_output_observed:
            raise ValueError("post-generation decision must observe one allowed model output")
        if not isinstance(self.raw_output, str):
            raise ValueError("post-generation decision requires raw_output")
        if self.action is PublicationAction.PUBLISH:
            if (
                not self.citation_syntax_passed
                or self.reason_code != "citation_syntax_passed"
                or self.response_text != self.raw_output
            ):
                raise ValueError("publish decision is inconsistent")
        elif self.action is PublicationAction.REJECT:
            if self.citation_syntax_passed or self.reason_code not in {
                "missing_citation",
                "unknown_citation",
                "uncited_paragraph",
            }:
                raise ValueError("post-generation reject decision is inconsistent")
        else:
            raise ValueError("post-generation decision must publish or reject")

    @property
    def decision_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    @property
    def public_decision_fingerprint(self) -> str:
        return _canonical_sha256(self.to_public_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        """Return the audit projection, including raw output and finding text."""

        projection: dict[str, object] = {
            "decision_version": RAG_PUBLICATION_POLICY_VERSION,
            "policy_revision": self.policy_revision,
            "stage": self.stage.value,
            "action": self.action.value,
            "reason_code": self.reason_code,
            "model_call_allowed": self.model_call_allowed,
            "generated_output_observed": self.generated_output_observed,
            "raw_output": self.raw_output,
            "raw_output_sha256": (
                _text_sha256(self.raw_output) if self.raw_output is not None else None
            ),
            "response_text": self.response_text,
            "response_text_sha256": _text_sha256(self.response_text),
            "valid_source_ids": list(self.valid_source_ids),
            "cited_source_ids": list(self.cited_source_ids),
            "unknown_source_ids": list(self.unknown_source_ids),
            "uncited_paragraphs": list(self.uncited_paragraphs),
            "citation_syntax_passed": self.citation_syntax_passed,
            "semantic_entailment_verified": self.semantic_entailment_verified,
        }
        if include_fingerprint:
            projection["decision_fingerprint"] = self.decision_fingerprint
        return projection

    def to_public_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        """Return a response projection that cannot disclose rejected raw output."""

        projection: dict[str, object] = {
            "decision_version": RAG_PUBLICATION_POLICY_VERSION,
            "policy_revision": self.policy_revision,
            "stage": self.stage.value,
            "action": self.action.value,
            "reason_code": self.reason_code,
            "response_text": self.response_text,
            "response_text_sha256": _text_sha256(self.response_text),
            "citation_syntax_passed": self.citation_syntax_passed,
            "semantic_entailment_verified": self.semantic_entailment_verified,
            "raw_output_included": False,
            "audit_findings_included": False,
        }
        if include_fingerprint:
            projection["public_decision_fingerprint"] = (
                self.public_decision_fingerprint
            )
        return projection


DEFAULT_RAG_PUBLICATION_POLICY = RAGPublicationPolicy()


@dataclass
class _RecordedOutputGenerator:
    expected_context: str
    raw_output: str
    calls: int = 0

    def __call__(self, rendered_context: str) -> str:
        self.calls += 1
        if rendered_context != self.expected_context:
            raise AssertionError("publication replay context drift")
        return self.raw_output


def evaluate_pre_generation(
    context: CitationContext,
    *,
    policy: RAGPublicationPolicy = DEFAULT_RAG_PUBLICATION_POLICY,
) -> RAGPublicationDecision | None:
    """Return a deterministic abstention only when authorized evidence is empty."""

    if context.sources:
        if not context.rendered.strip():
            raise ValueError("non-empty source map must have rendered context")
        return None
    if context.rendered:
        raise ValueError("empty source map must have empty rendered context")
    return RAGPublicationDecision(
        policy_revision=policy.revision,
        stage=PublicationStage.PRE_GENERATION,
        action=PublicationAction.ABSTAIN,
        reason_code="no_authorized_evidence",
        model_call_allowed=False,
        generated_output_observed=False,
        raw_output=None,
        response_text=policy.no_evidence_response,
        valid_source_ids=(),
        cited_source_ids=(),
        unknown_source_ids=(),
        uncited_paragraphs=(),
        citation_syntax_passed=False,
    )


def evaluate_post_generation(
    context: CitationContext,
    raw_output: str,
    *,
    policy: RAGPublicationPolicy = DEFAULT_RAG_PUBLICATION_POLICY,
) -> RAGPublicationDecision:
    """Publish only outputs that satisfy the local citation-syntax contract."""

    if not context.sources:
        raise ValueError("empty evidence must be handled before generation")
    if not isinstance(raw_output, str):
        raise TypeError("raw_output must be a string")
    if len(raw_output) > _MAX_RESPONSE_CHARACTERS:
        raise ValueError("raw_output exceeds the character limit")
    valid_source_ids = tuple(context.sources)
    audit = audit_citations(raw_output, valid_source_ids)
    citation_passed = bool(audit.cited_source_ids) and audit.syntactically_valid
    if citation_passed:
        action = PublicationAction.PUBLISH
        reason_code = "citation_syntax_passed"
        response_text = raw_output
    else:
        action = PublicationAction.REJECT
        if audit.unknown_source_ids:
            reason_code = "unknown_citation"
        elif not audit.cited_source_ids:
            reason_code = "missing_citation"
        else:
            reason_code = "uncited_paragraph"
        response_text = policy.rejected_response
    return RAGPublicationDecision(
        policy_revision=policy.revision,
        stage=PublicationStage.POST_GENERATION,
        action=action,
        reason_code=reason_code,
        model_call_allowed=True,
        generated_output_observed=True,
        raw_output=raw_output,
        response_text=response_text,
        valid_source_ids=valid_source_ids,
        cited_source_ids=audit.cited_source_ids,
        unknown_source_ids=audit.unknown_source_ids,
        uncited_paragraphs=audit.uncited_paragraphs,
        citation_syntax_passed=citation_passed,
    )


def guard_rag_generation(
    context: CitationContext,
    generate: Callable[[str], str],
    *,
    policy: RAGPublicationPolicy = DEFAULT_RAG_PUBLICATION_POLICY,
) -> RAGPublicationDecision:
    """Apply the pre-generation short-circuit and one post-generation gate."""

    early = evaluate_pre_generation(context, policy=policy)
    if early is not None:
        return early
    raw_output = generate(context.rendered)
    return evaluate_post_generation(context, raw_output, policy=policy)


def build_publication_policy_replay_report(
    *,
    spec: RAGTransformersControlSpec,
    source_report: Mapping[str, Any],
    policy: RAGPublicationPolicy = DEFAULT_RAG_PUBLICATION_POLICY,
) -> dict[str, object]:
    """Replay the policy over a separately verified real-model report."""

    if source_report.get("report_version") != RAG_TRANSFORMERS_REPORT_VERSION:
        raise ValueError("source report version is unsupported")
    if source_report.get("manifest_fingerprint") != spec.manifest_fingerprint:
        raise ValueError("source report does not bind the supplied control manifest")
    source_report_fingerprint = _sha256(
        source_report.get("report_fingerprint"), "source_report.report_fingerprint"
    )
    source_report_unsigned = dict(source_report)
    source_report_unsigned.pop("report_fingerprint", None)
    if _canonical_sha256(source_report_unsigned) != source_report_fingerprint:
        raise ValueError("source report fingerprint mismatch")
    source_cases = _array(source_report.get("cases"), "source_report.cases")
    if len(source_cases) != len(spec.cases):
        raise ValueError("source report case count differs from control manifest")
    documents = {item.document_id: item.to_document() for item in spec.corpus}
    case_reports: list[dict[str, object]] = []
    for index, (case_spec, raw_source_case) in enumerate(
        zip(spec.cases, source_cases, strict=True)
    ):
        location = f"source_report.cases[{index}]"
        source_case = _mapping(raw_source_case, location)
        if source_case.get("case_id") != case_spec.case_id:
            raise ValueError(f"{location}.case_id differs from control manifest")
        source_case_fingerprint = _sha256(
            source_case.get("case_fingerprint"), f"{location}.case_fingerprint"
        )
        source_case_unsigned = dict(source_case)
        source_case_unsigned.pop("case_fingerprint", None)
        if _canonical_sha256(source_case_unsigned) != source_case_fingerprint:
            raise ValueError(f"{location}.case_fingerprint mismatch")
        packing = _mapping(source_case.get("packing"), f"{location}.packing")
        packed_document_ids = _string_array(
            packing.get("document_ids"), f"{location}.packing.document_ids"
        )
        if packed_document_ids != list(case_spec.expected_packed_document_ids):
            raise ValueError(f"{location}.packing differs from control manifest")
        generation = _mapping(source_case.get("generation"), f"{location}.generation")
        raw_output = _plain_string(
            generation.get("raw_output"), f"{location}.generation.raw_output"
        )
        raw_output_sha256 = _sha256(
            generation.get("raw_output_sha256"),
            f"{location}.generation.raw_output_sha256",
        )
        if raw_output_sha256 != _text_sha256(raw_output):
            raise ValueError(f"{location}.generation raw output hash mismatch")
        verification = _mapping(
            source_case.get("verification"), f"{location}.verification"
        )
        baseline_gate = _boolean(
            verification.get("expected_behavior_gate_passed"),
            f"{location}.verification.expected_behavior_gate_passed",
        )
        packed_results = [
            SearchResult(
                document=documents[document_id],
                score=1.0,
                rank=rank,
                source="policy-replay",
            )
            for rank, document_id in enumerate(packed_document_ids, start=1)
        ]
        context = build_citation_context(
            packed_results,
            tenant_id=case_spec.tenant_id,
            principals=case_spec.principals,
        )
        recorded_generator = _RecordedOutputGenerator(context.rendered, raw_output)
        decision = guard_rag_generation(
            context, recorded_generator, policy=policy
        )
        expected_calls = int(bool(context.sources))
        if recorded_generator.calls != expected_calls:
            raise AssertionError("publication policy model-call ledger mismatch")
        case_projection: dict[str, object] = {
            "case_id": case_spec.case_id,
            "source_case_fingerprint": source_case_fingerprint,
            "baseline_raw_output_sha256": raw_output_sha256,
            "baseline_expected_behavior_gate_passed": baseline_gate,
            "packed_source_ids": list(context.sources),
            "policy_generator_call_count": recorded_generator.calls,
            "decision": decision.to_dict(),
        }
        case_projection["case_fingerprint"] = _canonical_sha256(case_projection)
        case_reports.append(case_projection)

    action_counts = {
        action.value: sum(
            cast(Mapping[str, Any], case["decision"])["action"] == action.value
            for case in case_reports
        )
        for action in PublicationAction
    }
    projection: dict[str, object] = {
        "report_version": RAG_PUBLICATION_REPLAY_REPORT_VERSION,
        "source_rag_report_fingerprint": source_report_fingerprint,
        "rag_control_manifest_fingerprint": spec.manifest_fingerprint,
        "policy": {
            **policy.to_dict(),
            "policy_fingerprint": policy.fingerprint,
        },
        "cases": case_reports,
        "summary": {
            "case_count": len(case_reports),
            "publish_count": action_counts[PublicationAction.PUBLISH.value],
            "pre_generation_abstention_count": action_counts[
                PublicationAction.ABSTAIN.value
            ],
            "post_generation_rejection_count": action_counts[
                PublicationAction.REJECT.value
            ],
            "unsafe_baseline_outputs_published_count": sum(
                (
                    cast(Mapping[str, Any], case["decision"])["action"]
                    == PublicationAction.PUBLISH.value
                    and case["baseline_expected_behavior_gate_passed"] is False
                )
                for case in case_reports
            ),
        },
        "scope": {
            "counterfactual_policy_replay_on_recorded_attempt": True,
            "raw_model_outputs_preserved_in_source_report": True,
            "no_evidence_model_call_would_be_suppressed": True,
            "invalid_citation_output_would_be_rejected": True,
            "guarded_runtime_model_call_suppression_observed": False,
            "claim_evidence_entailment_verified": False,
            "artifact_origin_authenticated_by_signature": False,
            "general_rag_quality_proven": False,
            "production_integration_proven": False,
        },
        "evidence_boundary": RAG_PUBLICATION_REPLAY_EVIDENCE_BOUNDARY,
    }
    projection["report_fingerprint"] = _canonical_sha256(projection)
    return projection


def verify_publication_policy_replay_report(
    path: Path,
    *,
    spec: RAGTransformersControlSpec,
    source_report: Mapping[str, Any],
    policy: RAGPublicationPolicy = DEFAULT_RAG_PUBLICATION_POLICY,
) -> Mapping[str, Any]:
    """Require exact deterministic reconstruction of every replay report field."""

    observed = _load_json_file(path)
    expected = build_publication_policy_replay_report(
        spec=spec, source_report=source_report, policy=policy
    )
    if observed != expected:
        raise ValueError(
            "publication policy replay report differs from deterministic reconstruction"
        )
    return observed


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return cast(Mapping[str, Any], value)


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _string_array(value: Any, location: str) -> list[str]:
    raw = _array(value, location)
    result: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{location}[{index}] must be a non-empty string")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{location} contains duplicate strings")
    return result


def _plain_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string")
    if len(value) > _MAX_RESPONSE_CHARACTERS:
        raise ValueError(f"{location} exceeds the character limit")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be a boolean")
    return value


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{location} must be a canonical SHA-256 fingerprint")
    return value


def _text_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return "sha256:" + artifact_fingerprint(cast(Mapping[str, object], value))


def _load_json_file(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{path}: cannot read JSON: {error}") from error
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"{path}: JSON exceeds the size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: JSON is not valid UTF-8") from error
    try:
        value: Any = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    canonical_json_bytes(value)
    return cast(dict[str, Any], value)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
