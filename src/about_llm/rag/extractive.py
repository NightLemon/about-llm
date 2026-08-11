"""Deterministic extractive answers over authorized, packed RAG evidence.

This is a deliberately lexical, non-LLM baseline.  It can prove that emitted
claims are exact substrings of the packed context; it cannot prove that the
source is true, that an answer is complete, or that lexical overlap implies
semantic relevance.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from about_llm.llmops import artifact_fingerprint
from about_llm.rag.answer_eval import (
    AnswerAction,
    ClaimVerdict,
    RecordedAnswer,
    RecordedClaim,
)
from about_llm.rag.context_packing import (
    ContextCost,
    PackedCitationContext,
    pack_citation_context,
)
from about_llm.rag.models import Document, SearchResult
from about_llm.rag.tokenization import lexical_tokens

EXTRACTIVE_ARTIFACT_VERSION = "about-llm.rag-extractive-answer.v1"
EXACT_SPAN_JUDGMENT = "deterministic-exact-source-span-v1"
_SENTENCE = re.compile(r"[^\n\u3002\uff01\uff1f!?\uff1b;.]+(?:[\u3002\uff01\uff1f!?\uff1b;.]|$)")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_QUERY_STOP_TOKENS = frozenset(
    {
        "rag",
        "的",
        "为",
        "什",
        "么",
        "是",
        "要",
        "做",
        "怎",
        "样",
        "不",
        "能",
        "又",
        "既",
        "把",
        "当",
        "成",
        "了",
        "吗",
        "呢",
        "请",
        "问",
        "哪",
        "些",
        "一",
        "个",
        "中",
        "和",
        "与",
        "或",
    }
)
_SCOPE_WARNING = (
    "this non-LLM baseline proves only exact-substring support inside authorized "
    "packed chunks; lexical coverage does not prove semantic relevance, source "
    "truth, answer completeness, or production safety"
)


@dataclass(frozen=True)
class ExtractiveAnswerConfig:
    """Visible lexical decision policy for the teaching baseline."""

    min_query_coverage: float = 0.55
    min_span_matched_tokens: int = 2
    max_answer_spans: int = 3
    max_spans_per_source: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_query_coverage) or not (
            0 < self.min_query_coverage <= 1
        ):
            raise ValueError("min_query_coverage must be finite and in (0, 1]")
        for name in (
            "min_span_matched_tokens",
            "max_answer_spans",
            "max_spans_per_source",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "min_query_coverage": self.min_query_coverage,
            "min_span_matched_tokens": self.min_span_matched_tokens,
            "max_answer_spans": self.max_answer_spans,
            "max_spans_per_source": self.max_spans_per_source,
            "query_token_policy": "distinct lexical tokens minus built-in stop tokens v1",
        }


@dataclass(frozen=True)
class ExtractiveSource:
    """Immutable source identity copied from one packed context."""

    short_id: str
    stable_source_id: str
    document_id: str
    source_version: str
    content_sha256: str
    retrieval_rank: int
    retrieval_score: float

    def __post_init__(self) -> None:
        for name in ("short_id", "stable_source_id", "document_id", "source_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if not _SHA256.fullmatch(self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase sha256 fingerprint")
        if self.retrieval_rank <= 0:
            raise ValueError("retrieval_rank must be positive")
        if not math.isfinite(self.retrieval_score):
            raise ValueError("retrieval_score must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "short_id": self.short_id,
            "stable_source_id": self.stable_source_id,
            "document_id": self.document_id,
            "source_version": self.source_version,
            "content_sha256": self.content_sha256,
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": self.retrieval_score,
        }


@dataclass(frozen=True)
class ExtractiveSpan:
    """An exact character span in one packed source document."""

    short_source_id: str
    stable_source_id: str
    document_id: str
    start_char: int
    end_char: int
    text: str
    matched_query_tokens: tuple[str, ...]
    newly_covered_query_tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "short_source_id": self.short_source_id,
            "stable_source_id": self.stable_source_id,
            "document_id": self.document_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
            "matched_query_tokens": list(self.matched_query_tokens),
            "newly_covered_query_tokens": list(self.newly_covered_query_tokens),
        }


@dataclass(frozen=True)
class ExtractiveAnswerArtifact:
    """Replayable output and proof ledger for one extractive answer attempt."""

    query_id: str
    query: str
    tenant_id: str
    principals: tuple[str, ...]
    config: ExtractiveAnswerConfig
    packed_context: PackedCitationContext
    sources: tuple[ExtractiveSource, ...]
    meaningful_query_tokens: tuple[str, ...]
    proposed_spans: tuple[ExtractiveSpan, ...]
    covered_query_tokens: tuple[str, ...]
    action: AnswerAction
    answer_text: str

    def __post_init__(self) -> None:
        for name in ("query_id", "query", "tenant_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if any(not principal.strip() for principal in self.principals):
            raise ValueError("principals cannot contain an empty value")
        if len(self.principals) != len(set(self.principals)):
            raise ValueError("principals cannot contain duplicates")
        if len(self.meaningful_query_tokens) != len(set(self.meaningful_query_tokens)):
            raise ValueError("meaningful_query_tokens must be unique")
        if any(not token for token in self.meaningful_query_tokens):
            raise ValueError("meaningful_query_tokens cannot contain empty values")
        expected_short_ids = tuple(self.packed_context.context.sources)
        if tuple(source.short_id for source in self.sources) != expected_short_ids:
            raise ValueError("source snapshots must match packed context order")
        source_by_short_id = {source.short_id: source for source in self.sources}
        if len(source_by_short_id) != len(self.sources):
            raise ValueError("source snapshots contain duplicate short ids")
        for source in self.sources:
            document = self.packed_context.context.sources[source.short_id]
            content_sha256 = "sha256:" + hashlib.sha256(
                document.text.encode("utf-8")
            ).hexdigest()
            if source.document_id != document.document_id:
                raise ValueError("source snapshot document identity does not match context")
            if source.stable_source_id != _stable_source_id(document):
                raise ValueError("source snapshot stable identity does not match context")
            if source.content_sha256 != content_sha256:
                raise ValueError("source snapshot content fingerprint does not match context")
        expected_covered: set[str] = set()
        for span in self.proposed_spans:
            span_source = source_by_short_id.get(span.short_source_id)
            if span_source is None:
                raise ValueError("extractive span cites a source outside packed context")
            document = self.packed_context.context.sources[span.short_source_id]
            if span_source.document_id != span.document_id:
                raise ValueError("extractive span document identity does not match source")
            if span_source.stable_source_id != span.stable_source_id:
                raise ValueError("extractive span stable source identity does not match")
            if not 0 <= span.start_char < span.end_char <= len(document.text):
                raise ValueError("extractive span character offsets are invalid")
            if document.text[span.start_char : span.end_char] != span.text:
                raise ValueError("extractive span is not an exact source substring")
            exact_matches = tuple(
                token
                for token in self.meaningful_query_tokens
                if token in set(lexical_tokens(span.text))
            )
            if span.matched_query_tokens != exact_matches:
                raise ValueError("span matched_query_tokens do not match lexical evidence")
            expected_new = tuple(
                token for token in exact_matches if token not in expected_covered
            )
            if span.newly_covered_query_tokens != expected_new:
                raise ValueError("span newly covered token ledger is inconsistent")
            expected_covered.update(exact_matches)
        expected_covered_order = tuple(
            token for token in self.meaningful_query_tokens if token in expected_covered
        )
        if self.covered_query_tokens != expected_covered_order:
            raise ValueError("covered_query_tokens do not match proposed spans")
        expected_answer = (
            "\n".join(
                f"{span.text} [{span.short_source_id}]" for span in self.proposed_spans
            )
            if self.action is AnswerAction.ANSWER
            else "证据不足，无法基于已授权知识库回答。"  # noqa: RUF001
        )
        if self.answer_text != expected_answer:
            raise ValueError("answer_text does not match the deterministic rendering")
        passes = self.coverage >= self.config.min_query_coverage
        if self.action is AnswerAction.ANSWER and (not self.proposed_spans or not passes):
            raise ValueError("answer action requires spans that pass the coverage threshold")
        if self.action is AnswerAction.ABSTAIN and passes:
            raise ValueError("abstain action cannot pass the coverage threshold")
        if self.action is AnswerAction.ERROR:
            raise ValueError("extractive baseline does not encode runtime errors as artifacts")

    @property
    def coverage(self) -> float:
        if not self.meaningful_query_tokens:
            return 0.0
        return len(self.covered_query_tokens) / len(self.meaningful_query_tokens)

    @property
    def recorded_answer(self) -> RecordedAnswer:
        context_source_ids = tuple(
            dict.fromkeys(source.stable_source_id for source in self.sources)
        )
        if self.action is AnswerAction.ANSWER:
            claims = tuple(
                RecordedClaim(
                    claim_id=f"extractive-span-{index}",
                    text=span.text,
                    source_ids=(span.stable_source_id,),
                    verdict=ClaimVerdict.SUPPORTED,
                    judgment_source=EXACT_SPAN_JUDGMENT,
                )
                for index, span in enumerate(self.proposed_spans, start=1)
            )
            missing_information: tuple[str, ...] = ()
        else:
            claims = ()
            missing_information = (
                "授权后的已打包证据未达到 lexical coverage threshold",
            )
        return RecordedAnswer(
            query_id=self.query_id,
            action=self.action,
            context_source_ids=context_source_ids,
            claims=claims,
            missing_information=missing_information,
        )

    @property
    def artifact_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self.identity_dict())

    def identity_dict(self) -> dict[str, object]:
        return {
            "artifact_version": EXTRACTIVE_ARTIFACT_VERSION,
            "query_id": self.query_id,
            "query_sha256": "sha256:"
            + hashlib.sha256(self.query.encode("utf-8")).hexdigest(),
            "tenant_id": self.tenant_id,
            "principals": list(self.principals),
            "config": self.config.to_dict(),
            "packing": {
                "budget_units": self.packed_context.budget_units,
                "base_cost_units": self.packed_context.base_cost_units,
                "used_cost_units": self.packed_context.used_cost_units,
                "cost_unit": self.packed_context.cost_unit,
                "max_chunks_per_source": self.packed_context.max_chunks_per_source,
                "decisions": [
                    {
                        "document_id": decision.document_id,
                        "stable_source_id": decision.stable_source_id,
                        "rank": decision.rank,
                        "selected": decision.selected,
                        "reason": decision.reason.value,
                        "cost_if_selected_units": decision.cost_if_selected_units,
                    }
                    for decision in self.packed_context.decisions
                ],
            },
            "rendered_context": self.packed_context.context.rendered,
            "sources": [source.to_dict() for source in self.sources],
            "meaningful_query_tokens": list(self.meaningful_query_tokens),
            "proposed_spans": [span.to_dict() for span in self.proposed_spans],
            "covered_query_tokens": list(self.covered_query_tokens),
            "coverage": self.coverage,
            "action": self.action.value,
            "answer_text": self.answer_text,
            "recorded_answer": self.recorded_answer.to_dict(),
            "scope_warning": _SCOPE_WARNING,
        }

    def to_dict(self) -> dict[str, object]:
        value = self.identity_dict()
        value["artifact_fingerprint"] = self.artifact_fingerprint
        return value


@dataclass(frozen=True)
class _CandidateSpan:
    short_source_id: str
    stable_source_id: str
    document_id: str
    start_char: int
    end_char: int
    text: str
    matched_query_tokens: tuple[str, ...]
    retrieval_rank: int


def generate_extractive_answer(
    results: Iterable[SearchResult],
    *,
    query_id: str,
    query: str,
    tenant_id: str,
    cost_fn: ContextCost,
    budget_units: int,
    cost_unit: str,
    principals: Iterable[str] = (),
    max_chunks_per_source: int = 2,
    config: ExtractiveAnswerConfig | None = None,
) -> ExtractiveAnswerArtifact:
    """Pack authorized results, then copy enough exact spans to pass a lexical gate.

    No qrels, answerability labels, or expected source ids are accepted by this
    API, so offline labels cannot directly control the online action.
    """
    if not query_id.strip() or not query.strip() or not tenant_id.strip():
        raise ValueError("query_id, query, and tenant_id must be non-empty")
    principal_tuple = tuple(principals)
    candidate_results = tuple(results)
    packed = pack_citation_context(
        candidate_results,
        tenant_id=tenant_id,
        principals=principal_tuple,
        budget_units=budget_units,
        cost_fn=cost_fn,
        cost_unit=cost_unit,
        max_chunks_per_source=max_chunks_per_source,
    )
    policy = config or ExtractiveAnswerConfig()
    meaningful_tokens = _meaningful_query_tokens(query)
    result_by_document: dict[str, SearchResult] = {}
    for result in candidate_results:
        result_by_document.setdefault(result.document.document_id, result)
    sources = tuple(
        _source_snapshot(short_id, document, result_by_document)
        for short_id, document in packed.context.sources.items()
    )
    candidates = _candidate_spans(
        packed,
        sources,
        meaningful_tokens,
        min_matched_tokens=policy.min_span_matched_tokens,
    )
    proposed, covered = _greedy_cover(candidates, meaningful_tokens, policy)
    coverage = len(covered) / len(meaningful_tokens) if meaningful_tokens else 0.0
    action = (
        AnswerAction.ANSWER
        if proposed and coverage >= policy.min_query_coverage
        else AnswerAction.ABSTAIN
    )
    rendered_answer = (
        "\n".join(f"{span.text} [{span.short_source_id}]" for span in proposed)
        if action is AnswerAction.ANSWER
        else "证据不足，无法基于已授权知识库回答。"  # noqa: RUF001
    )
    return ExtractiveAnswerArtifact(
        query_id=query_id,
        query=query,
        tenant_id=tenant_id,
        principals=principal_tuple,
        config=policy,
        packed_context=packed,
        sources=sources,
        meaningful_query_tokens=meaningful_tokens,
        proposed_spans=proposed,
        covered_query_tokens=tuple(token for token in meaningful_tokens if token in covered),
        action=action,
        answer_text=rendered_answer,
    )


def _meaningful_query_tokens(query: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token for token in lexical_tokens(query) if token not in _QUERY_STOP_TOKENS
        )
    )


def _source_snapshot(
    short_id: str,
    document: Document,
    result_by_document: dict[str, SearchResult],
) -> ExtractiveSource:
    result = result_by_document[document.document_id]
    stable_source_id = _metadata_string(document, "source_id", document.document_id)
    source_version = _metadata_string(document, "source_version", "unknown")
    return ExtractiveSource(
        short_id=short_id,
        stable_source_id=stable_source_id,
        document_id=document.document_id,
        source_version=source_version,
        content_sha256="sha256:"
        + hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
        retrieval_rank=result.rank,
        retrieval_score=result.score,
    )


def _candidate_spans(
    packed: PackedCitationContext,
    sources: Sequence[ExtractiveSource],
    meaningful_query_tokens: tuple[str, ...],
    *,
    min_matched_tokens: int,
) -> tuple[_CandidateSpan, ...]:
    source_by_short_id = {source.short_id: source for source in sources}
    candidates: list[_CandidateSpan] = []
    for short_id, document in packed.context.sources.items():
        source = source_by_short_id[short_id]
        for match in _SENTENCE.finditer(document.text):
            raw = match.group(0)
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw) - len(raw.rstrip())
            start = match.start() + leading
            end = match.end() - trailing
            if start >= end:
                continue
            text = document.text[start:end]
            span_tokens = set(lexical_tokens(text))
            matched = tuple(token for token in meaningful_query_tokens if token in span_tokens)
            if len(matched) < min_matched_tokens:
                continue
            candidates.append(
                _CandidateSpan(
                    short_source_id=short_id,
                    stable_source_id=source.stable_source_id,
                    document_id=document.document_id,
                    start_char=start,
                    end_char=end,
                    text=text,
                    matched_query_tokens=matched,
                    retrieval_rank=source.retrieval_rank,
                )
            )
    return tuple(candidates)


def _greedy_cover(
    candidates: Sequence[_CandidateSpan],
    query_tokens: tuple[str, ...],
    config: ExtractiveAnswerConfig,
) -> tuple[tuple[ExtractiveSpan, ...], set[str]]:
    selected: list[ExtractiveSpan] = []
    covered: set[str] = set()
    remaining = list(candidates)
    per_source: dict[str, int] = {}
    query_token_set = set(query_tokens)
    while remaining and len(selected) < config.max_answer_spans:
        eligible = [
            candidate
            for candidate in remaining
            if per_source.get(candidate.stable_source_id, 0)
            < config.max_spans_per_source
            and set(candidate.matched_query_tokens) - covered
        ]
        if not eligible:
            break
        best = min(
            eligible,
            key=lambda candidate: (
                -len(set(candidate.matched_query_tokens) - covered),
                -(candidate.stable_source_id not in per_source),
                -len(candidate.matched_query_tokens),
                candidate.retrieval_rank,
                candidate.start_char,
                candidate.document_id,
            ),
        )
        newly_covered = set(best.matched_query_tokens) - covered
        new_tokens = tuple(token for token in query_tokens if token in newly_covered)
        selected.append(
            ExtractiveSpan(
                short_source_id=best.short_source_id,
                stable_source_id=best.stable_source_id,
                document_id=best.document_id,
                start_char=best.start_char,
                end_char=best.end_char,
                text=best.text,
                matched_query_tokens=best.matched_query_tokens,
                newly_covered_query_tokens=new_tokens,
            )
        )
        covered.update(best.matched_query_tokens)
        per_source[best.stable_source_id] = per_source.get(best.stable_source_id, 0) + 1
        remaining.remove(best)
        if query_token_set and len(covered) / len(query_token_set) >= config.min_query_coverage:
            break
    return tuple(selected), covered


def _metadata_string(document: Document, key: str, default: str) -> str:
    value = document.metadata.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"document {document.document_id!r} has invalid {key!r} metadata")
    return value


def _stable_source_id(document: Document) -> str:
    return _metadata_string(document, "source_id", document.document_id)
