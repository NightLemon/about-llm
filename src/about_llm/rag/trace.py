"""Strict RAG generation traces binding packed evidence to recorded answers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes
from about_llm.rag.answer_eval import RecordedAnswer
from about_llm.rag.citations import build_citation_context
from about_llm.rag.models import Document, SearchResult

RAG_GENERATION_TRACE_VERSION = "about-llm.rag-generation-trace.v1"
RAG_GENERATION_TRACE_AUDIT_VERSION = "about-llm.rag-generation-trace-audit.v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class RAGTraceCaseBinding:
    query_sha256: str
    tenant_id: str
    principals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _fingerprint(self.query_sha256, "query_sha256")
        _nonempty(self.tenant_id, "tenant_id")
        _unique_strings(self.principals, "principals")


@dataclass(frozen=True)
class RAGTraceSource:
    short_id: str
    stable_source_id: str
    document_id: str
    source_version: str
    content_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("short_id", self.short_id),
            ("stable_source_id", self.stable_source_id),
            ("document_id", self.document_id),
            ("source_version", self.source_version),
        ):
            _nonempty(value, name)
        _fingerprint(self.content_sha256, "content_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "short_id": self.short_id,
            "stable_source_id": self.stable_source_id,
            "document_id": self.document_id,
            "source_version": self.source_version,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class RAGGenerationTrace:
    trace_id: str
    query_id: str
    query_sha256: str
    tenant_id: str
    principals: tuple[str, ...]
    sources: tuple[RAGTraceSource, ...]
    rendered_context: str
    prompt_token_ids: tuple[int, ...]
    tokenizer_revision: str
    chat_template_sha256: str
    system_prompt_sha256: str
    user_prompt_template_sha256: str
    reserved_output_tokens: int
    generator_revision: str
    raw_output: str
    recorded_answer_fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("trace_id", self.trace_id),
            ("query_id", self.query_id),
            ("tenant_id", self.tenant_id),
            ("tokenizer_revision", self.tokenizer_revision),
            ("generator_revision", self.generator_revision),
        ):
            _nonempty(value, name)
        _unique_strings(self.principals, "principals")
        _fingerprint(self.query_sha256, "query_sha256")
        if [source.short_id for source in self.sources] != [
            f"S{index}" for index in range(1, len(self.sources) + 1)
        ]:
            raise ValueError("trace source short_id values must be ordered S1..Sn")
        document_ids = [source.document_id for source in self.sources]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("trace sources contain duplicate document_id")
        if not isinstance(self.rendered_context, str):
            raise TypeError("rendered_context must be a string")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in self.prompt_token_ids
        ):
            raise ValueError("prompt_token_ids must be non-negative integers")
        for name, value in (
            ("chat_template_sha256", self.chat_template_sha256),
            ("system_prompt_sha256", self.system_prompt_sha256),
            ("user_prompt_template_sha256", self.user_prompt_template_sha256),
            ("recorded_answer_fingerprint", self.recorded_answer_fingerprint),
        ):
            _fingerprint(value, name)
        if (
            isinstance(self.reserved_output_tokens, bool)
            or not isinstance(self.reserved_output_tokens, int)
            or self.reserved_output_tokens <= 0
        ):
            raise ValueError("reserved_output_tokens must be a positive integer")
        if not isinstance(self.raw_output, str):
            raise TypeError("raw_output must be a string")
        snapshot = json.loads(canonical_json_bytes(self.metadata))
        if not isinstance(snapshot, dict):
            raise ValueError("metadata must be a JSON object")
        object.__setattr__(self, "metadata", _freeze(snapshot))

    @property
    def context_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(
            {
                "sources": [source.to_dict() for source in self.sources],
                "rendered_context": self.rendered_context,
            }
        )

    @property
    def prompt_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(
            {
                "query_sha256": self.query_sha256,
                "prompt_token_ids": list(self.prompt_token_ids),
                "tokenizer_revision": self.tokenizer_revision,
                "chat_template_sha256": self.chat_template_sha256,
                "system_prompt_sha256": self.system_prompt_sha256,
                "user_prompt_template_sha256": self.user_prompt_template_sha256,
                "reserved_output_tokens": self.reserved_output_tokens,
            }
        )

    @property
    def raw_output_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.raw_output.encode("utf-8")).hexdigest()

    @property
    def trace_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_version": RAG_GENERATION_TRACE_VERSION,
            "trace_id": self.trace_id,
            "query_id": self.query_id,
            "query_sha256": self.query_sha256,
            "tenant_id": self.tenant_id,
            "principals": list(self.principals),
            "sources": [source.to_dict() for source in self.sources],
            "rendered_context": self.rendered_context,
            "prompt_token_ids": list(self.prompt_token_ids),
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_sha256": self.chat_template_sha256,
            "system_prompt_sha256": self.system_prompt_sha256,
            "user_prompt_template_sha256": self.user_prompt_template_sha256,
            "reserved_output_tokens": self.reserved_output_tokens,
            "generator_revision": self.generator_revision,
            "raw_output": self.raw_output,
            "recorded_answer_fingerprint": self.recorded_answer_fingerprint,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True)
class RAGTraceFinding:
    query_id: str
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"query_id": self.query_id, "code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class RAGTraceAuditReport:
    expected_query_ids: tuple[str, ...]
    trace_count: int
    answer_count: int
    finding_counts: Mapping[str, int]
    findings: tuple[RAGTraceFinding, ...]
    trace_summaries: tuple[Mapping[str, object], ...]

    @property
    def gate_passed(self) -> bool:
        return not self.findings

    @property
    def manifest_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(
            {
                "audit_version": RAG_GENERATION_TRACE_AUDIT_VERSION,
                "expected_query_ids": list(self.expected_query_ids),
                "trace_fingerprints": [
                    summary["trace_fingerprint"] for summary in self.trace_summaries
                ],
                "answer_fingerprints": [
                    summary["recorded_answer_fingerprint"]
                    for summary in self.trace_summaries
                ],
                "finding_counts": dict(self.finding_counts),
                "findings": [finding.to_dict() for finding in self.findings],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_version": RAG_GENERATION_TRACE_AUDIT_VERSION,
            "trace_version": RAG_GENERATION_TRACE_VERSION,
            "gate_passed": self.gate_passed,
            "expected_query_ids": list(self.expected_query_ids),
            "trace_count": self.trace_count,
            "answer_count": self.answer_count,
            "finding_counts": dict(self.finding_counts),
            "findings": [finding.to_dict() for finding in self.findings],
            "trace_summaries": [dict(summary) for summary in self.trace_summaries],
            "manifest_fingerprint": self.manifest_fingerprint,
            "scope": {
                "trace_query_and_security_context_verified": self.gate_passed,
                "trace_answer_canonical_identity_verified": self.gate_passed,
                "trace_context_reconstructed_from_current_chunks": self.gate_passed,
                "raw_output_claim_semantics_verified": False,
                "remote_model_execution_verified": False,
                "artifact_origin_cryptographically_authenticated": False,
                "historical_corpus_availability_proven": False,
            },
            "evidence_boundary": (
                "A passing audit binds the supplied query hash and security context to current "
                "chunk bytes, canonical rendered context, supplied prompt identity, raw output "
                "bytes, and the structured RecordedAnswer identity. It does not retokenize the "
                "prompt, compare revisions with a trusted registry, prove a remote model "
                "executed, prove raw output semantically entails parsed claims, establish that "
                "judgments are correct, or authenticate unsigned file provenance."
            ),
        }


def load_rag_generation_traces(path: Path) -> tuple[RAGGenerationTrace, ...]:
    traces: list[RAGGenerationTrace] = []
    trace_ids: set[str] = set()
    query_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        location = f"{path}:{line_number}"
        try:
            value: Any = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{location}: invalid strict JSON: {error}") from error
        record = _object(value, location)
        fields = {
            "trace_version",
            "trace_id",
            "query_id",
            "query_sha256",
            "tenant_id",
            "principals",
            "sources",
            "rendered_context",
            "prompt_token_ids",
            "tokenizer_revision",
            "chat_template_sha256",
            "system_prompt_sha256",
            "user_prompt_template_sha256",
            "reserved_output_tokens",
            "generator_revision",
            "raw_output",
            "recorded_answer_fingerprint",
            "metadata",
        }
        _fields(record, required=fields - {"metadata"}, allowed=fields, location=location)
        trace_version = _string(
            record["trace_version"], f"{location}.trace_version"
        )
        if trace_version != RAG_GENERATION_TRACE_VERSION:
            raise ValueError(
                f"{location}.trace_version: expected "
                f"{RAG_GENERATION_TRACE_VERSION!r}, got {trace_version!r}"
            )
        raw_sources = _array(record["sources"], f"{location}.sources")
        sources: list[RAGTraceSource] = []
        for index, raw_source in enumerate(raw_sources):
            source_location = f"{location}.sources[{index}]"
            source = _object(raw_source, source_location)
            source_fields = {
                "short_id",
                "stable_source_id",
                "document_id",
                "source_version",
                "content_sha256",
            }
            _fields(
                source,
                required=source_fields,
                allowed=source_fields,
                location=source_location,
            )
            sources.append(
                RAGTraceSource(
                    short_id=_string(source["short_id"], f"{source_location}.short_id"),
                    stable_source_id=_string(
                        source["stable_source_id"],
                        f"{source_location}.stable_source_id",
                    ),
                    document_id=_string(
                        source["document_id"], f"{source_location}.document_id"
                    ),
                    source_version=_string(
                        source["source_version"],
                        f"{source_location}.source_version",
                    ),
                    content_sha256=_string(
                        source["content_sha256"],
                        f"{source_location}.content_sha256",
                    ),
                )
            )
        trace = RAGGenerationTrace(
            trace_id=_string(record["trace_id"], f"{location}.trace_id"),
            query_id=_string(record["query_id"], f"{location}.query_id"),
            query_sha256=_string(
                record["query_sha256"], f"{location}.query_sha256"
            ),
            tenant_id=_string(record["tenant_id"], f"{location}.tenant_id"),
            principals=_strings(record["principals"], f"{location}.principals"),
            sources=tuple(sources),
            rendered_context=_plain_string(
                record["rendered_context"], f"{location}.rendered_context"
            ),
            prompt_token_ids=_integers(
                record["prompt_token_ids"], f"{location}.prompt_token_ids"
            ),
            tokenizer_revision=_string(
                record["tokenizer_revision"], f"{location}.tokenizer_revision"
            ),
            chat_template_sha256=_string(
                record["chat_template_sha256"], f"{location}.chat_template_sha256"
            ),
            system_prompt_sha256=_string(
                record["system_prompt_sha256"], f"{location}.system_prompt_sha256"
            ),
            user_prompt_template_sha256=_string(
                record["user_prompt_template_sha256"],
                f"{location}.user_prompt_template_sha256",
            ),
            reserved_output_tokens=_integer(
                record["reserved_output_tokens"],
                f"{location}.reserved_output_tokens",
            ),
            generator_revision=_string(
                record["generator_revision"], f"{location}.generator_revision"
            ),
            raw_output=_plain_string(record["raw_output"], f"{location}.raw_output"),
            recorded_answer_fingerprint=_string(
                record["recorded_answer_fingerprint"],
                f"{location}.recorded_answer_fingerprint",
            ),
            metadata=record.get("metadata", {}),
        )
        if trace.trace_id in trace_ids:
            raise ValueError(f"{location}: duplicate trace_id {trace.trace_id!r}")
        if trace.query_id in query_ids:
            raise ValueError(f"{location}: duplicate query_id {trace.query_id!r}")
        trace_ids.add(trace.trace_id)
        query_ids.add(trace.query_id)
        traces.append(trace)
    if not traces:
        raise ValueError(f"{path}: trace dataset contains no records")
    return tuple(traces)


def audit_rag_generation_traces(
    *,
    expected_cases: Mapping[str, RAGTraceCaseBinding],
    answers: Sequence[RecordedAnswer],
    traces: Sequence[RAGGenerationTrace],
    documents: Iterable[Document],
) -> RAGTraceAuditReport:
    if not expected_cases:
        raise ValueError("expected_cases must not be empty")
    answer_by_id = {answer.query_id: answer for answer in answers}
    trace_by_id = {trace.query_id: trace for trace in traces}
    if len(answer_by_id) != len(answers):
        raise ValueError("answers contain duplicate query ids")
    if len(trace_by_id) != len(traces):
        raise ValueError("traces contain duplicate query ids")
    document_snapshot = tuple(documents)
    document_by_id = {document.document_id: document for document in document_snapshot}
    if len(document_by_id) != len(document_snapshot):
        raise ValueError("documents contain duplicate document ids")

    findings: list[RAGTraceFinding] = []
    expected_ids = tuple(expected_cases)
    for label, actual in (("answer", answer_by_id), ("trace", trace_by_id)):
        for query_id in sorted(set(expected_cases) - set(actual)):
            findings.append(RAGTraceFinding(query_id, f"missing_{label}", label))
        for query_id in sorted(set(actual) - set(expected_cases)):
            findings.append(RAGTraceFinding(query_id, f"extra_{label}", label))

    summaries: list[Mapping[str, object]] = []
    for query_id in expected_ids:
        answer = answer_by_id.get(query_id)
        trace = trace_by_id.get(query_id)
        if answer is None or trace is None:
            continue
        binding = expected_cases[query_id]
        if trace.query_sha256 != binding.query_sha256:
            findings.append(
                RAGTraceFinding(query_id, "query_fingerprint_mismatch", "canonical query")
            )
        if trace.tenant_id != binding.tenant_id:
            findings.append(RAGTraceFinding(query_id, "tenant_mismatch", trace.tenant_id))
        if trace.principals != binding.principals:
            findings.append(
                RAGTraceFinding(query_id, "principals_mismatch", repr(trace.principals))
            )
        if trace.recorded_answer_fingerprint != answer.record_fingerprint:
            findings.append(
                RAGTraceFinding(query_id, "answer_fingerprint_mismatch", "canonical answer")
            )
        stable_ids = tuple(
            dict.fromkeys(source.stable_source_id for source in trace.sources)
        )
        if stable_ids != answer.context_source_ids:
            findings.append(
                RAGTraceFinding(query_id, "context_source_order_mismatch", repr(stable_ids))
            )

        resolved: list[Document] = []
        can_reconstruct = True
        principal_set = set(trace.principals)
        for source in trace.sources:
            document = document_by_id.get(source.document_id)
            if document is None:
                findings.append(
                    RAGTraceFinding(query_id, "unknown_document", source.document_id)
                )
                can_reconstruct = False
                continue
            resolved.append(document)
            actual_source = document.metadata.get("source_id")
            actual_version = document.metadata.get("source_version")
            actual_hash = "sha256:" + hashlib.sha256(
                document.text.encode("utf-8")
            ).hexdigest()
            if document.tenant_id != trace.tenant_id:
                findings.append(
                    RAGTraceFinding(query_id, "document_tenant_mismatch", source.document_id)
                )
            if actual_source != source.stable_source_id:
                findings.append(
                    RAGTraceFinding(query_id, "source_identity_mismatch", source.document_id)
                )
            if actual_version != source.source_version:
                findings.append(
                    RAGTraceFinding(query_id, "source_version_mismatch", source.document_id)
                )
            if actual_hash != source.content_sha256:
                findings.append(
                    RAGTraceFinding(query_id, "source_content_mismatch", source.document_id)
                )
            if document.acl and not principal_set.intersection(document.acl):
                findings.append(
                    RAGTraceFinding(query_id, "source_not_authorized", source.document_id)
                )
        if can_reconstruct:
            try:
                context = build_citation_context(
                    [
                        SearchResult(document, 0.0, index, "trace-reconstruction")
                        for index, document in enumerate(resolved, 1)
                    ],
                    tenant_id=trace.tenant_id,
                    principals=trace.principals,
                )
            except PermissionError as error:
                findings.append(
                    RAGTraceFinding(query_id, "context_reconstruction_denied", str(error))
                )
            else:
                if context.rendered != trace.rendered_context:
                    findings.append(
                        RAGTraceFinding(query_id, "rendered_context_mismatch", "canonical")
                    )
                if tuple(context.sources) != tuple(
                    source.short_id for source in trace.sources
                ):
                    findings.append(
                        RAGTraceFinding(query_id, "short_source_map_mismatch", "canonical")
                    )
        summaries.append(
            MappingProxyType(
                {
                    "query_id": query_id,
                    "trace_id": trace.trace_id,
                    "trace_fingerprint": trace.trace_fingerprint,
                    "context_fingerprint": trace.context_fingerprint,
                    "prompt_fingerprint": trace.prompt_fingerprint,
                    "raw_output_sha256": trace.raw_output_sha256,
                    "recorded_answer_fingerprint": answer.record_fingerprint,
                    "source_count": len(trace.sources),
                    "prompt_token_count": len(trace.prompt_token_ids),
                }
            )
        )
    counts = Counter(finding.code for finding in findings)
    return RAGTraceAuditReport(
        expected_query_ids=expected_ids,
        trace_count=len(traces),
        answer_count=len(answers),
        finding_counts=MappingProxyType(dict(sorted(counts.items()))),
        findings=tuple(findings),
        trace_summaries=tuple(summaries),
    )


def _nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _unique_strings(values: Sequence[str], name: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


def _fingerprint(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")


def _freeze(value: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            key: _freeze(item) if isinstance(item, dict) else _freeze_list(item)
            for key, item in value.items()
        }
    )


def _freeze_list(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(
            _freeze(item) if isinstance(item, dict) else _freeze_list(item)
            for item in value
        )
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: expected an object")
    return cast(dict[str, Any], value)


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location}: expected an array")
    return value


def _fields(
    value: Mapping[str, Any], *, required: set[str], allowed: set[str], location: str
) -> None:
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing or unknown:
        raise ValueError(
            f"{location}: field mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _string(value: Any, location: str) -> str:
    _nonempty(value, location)
    return cast(str, value)


def _plain_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location}: expected a string")
    return value


def _strings(value: Any, location: str) -> tuple[str, ...]:
    items = _array(value, location)
    result = tuple(_string(item, location) for item in items)
    _unique_strings(result, location)
    return result


def _integers(value: Any, location: str) -> tuple[int, ...]:
    items = _array(value, location)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in items):
        raise ValueError(f"{location}: expected integer token ids")
    return tuple(cast(list[int], items))


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location}: expected an integer")
    return cast(int, value)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
