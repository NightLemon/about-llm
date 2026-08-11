"""Runnable RAG baseline with optional target-tokenizer context packing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from about_llm.evaluation import (
    all_evidence_recall_at_k,
    mean_reciprocal_rank,
    normalized_discounted_cumulative_gain,
    precision_at_k,
    recall_at_k,
)
from about_llm.rag.answer_eval import (
    RecordedAnswer,
    evaluate_recorded_answers,
    load_recorded_answers,
)
from about_llm.rag.bm25 import BM25Index
from about_llm.rag.citations import audit_citations, build_citation_context
from about_llm.rag.context_packing import (
    make_rag_chat_prompt_cost,
    pack_citation_context,
    utf8_byte_length,
)
from about_llm.rag.extractive import (
    ExtractiveAnswerArtifact,
    ExtractiveAnswerConfig,
    generate_extractive_answer,
)
from about_llm.rag.ingestion import SourceChunk, SourceDocument, split_markdown
from about_llm.rag.models import Document, SearchResult
from about_llm.rag.reranking import (
    RecordedRerankScore,
    RecordedRerankScorer,
    rerank_authorized_candidates,
)
from about_llm.rag.sqlite_backup import (
    SQLiteChunkBackupManifest,
    create_sqlite_chunk_backup,
    restore_sqlite_chunk_backup,
    verify_sqlite_chunk_backup,
)
from about_llm.rag.sqlite_store import SQLiteChunkStore
from about_llm.rag.trace import (
    RAGTraceCaseBinding,
    audit_rag_generation_traces,
    load_rag_generation_traces,
)


@dataclass(frozen=True)
class RetrievalCase:
    """One source-level case with graded labels and a security context."""

    query_id: str
    query: str
    tenant_id: str
    relevant_source_ids: frozenset[str]
    principals: tuple[str, ...] = ()
    relevance: Mapping[str, float] = field(default_factory=dict)
    required_source_ids: frozenset[str] = frozenset()
    answerable: bool = True

    def __post_init__(self) -> None:
        labels = dict(self.relevance)
        if not labels and self.relevant_source_ids:
            labels = {source_id: 1.0 for source_id in self.relevant_source_ids}
        if any(not source_id.strip() for source_id in labels):
            raise ValueError("relevance source ids cannot be empty")
        if any(not math.isfinite(value) or value < 0 for value in labels.values()):
            raise ValueError("relevance values must be finite and non-negative")
        positive_ids = frozenset(
            source_id for source_id, value in labels.items() if value > 0
        )
        if self.relevant_source_ids and self.relevant_source_ids != positive_ids:
            raise ValueError("relevant_source_ids must match positive relevance labels")
        required = self.required_source_ids or positive_ids
        if not required <= positive_ids:
            raise ValueError("required_source_ids must have positive relevance labels")
        if self.answerable and not positive_ids:
            raise ValueError("an answerable case needs positive relevance labels")
        if not self.answerable and (positive_ids or required):
            raise ValueError("a no-answer case cannot have positive or required evidence")
        object.__setattr__(self, "relevant_source_ids", positive_ids)
        object.__setattr__(self, "relevance", MappingProxyType(labels))
        object.__setattr__(self, "required_source_ids", required)


class MarkdownBM25Pipeline:
    """In-memory Markdown ingestion, BM25 retrieval, ACL, and citations."""

    def __init__(self, sources: Iterable[SourceDocument], *, max_chars: int = 1200) -> None:
        chunks = tuple(
            chunk for source in sources for chunk in split_markdown(source, max_chars=max_chars)
        )
        if not chunks:
            raise ValueError("corpus produced no indexable chunks")
        self.chunks = chunks
        self.documents = tuple(_chunk_to_document(chunk) for chunk in chunks)
        self.index = BM25Index(self.documents)

    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        principals: Iterable[str] = (),
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Retrieve only chunks visible to the tenant and caller principals."""
        return self.index.search(
            query,
            tenant_id=tenant_id,
            principals=principals,
            top_k=top_k,
        )


def load_corpus(path: Path) -> tuple[SourceDocument, ...]:
    """Load versioned Markdown sources from a UTF-8 JSONL corpus."""
    records = _read_jsonl(path)
    sources: list[SourceDocument] = []
    identities: set[tuple[str, str]] = set()
    for line_number, record in records:
        source_id = _required_string(record, "source_id", path, line_number)
        tenant_id = _required_string(record, "tenant_id", path, line_number)
        identity = tenant_id, source_id
        if identity in identities:
            raise ValueError(
                f"{path}:{line_number}: duplicate tenant/source identity {identity!r}"
            )
        identities.add(identity)
        acl = _string_list(record.get("acl", []), "acl", path, line_number)
        metadata_value = record.get("metadata", {})
        if not isinstance(metadata_value, dict) or not all(
            isinstance(key, str) for key in metadata_value
        ):
            raise ValueError(f"{path}:{line_number}: metadata must be a JSON object")
        sources.append(
            SourceDocument(
                source_id=source_id,
                tenant_id=tenant_id,
                version=_required_string(record, "version", path, line_number),
                text=_required_string(record, "text", path, line_number),
                acl=acl,
                metadata=cast(dict[str, Any], metadata_value),
            )
        )
    if not sources:
        raise ValueError(f"{path}: corpus must contain at least one non-empty JSON line")
    return tuple(sources)


def load_cases(path: Path) -> tuple[RetrievalCase, ...]:
    """Load source-level binary/graded labels and no-answer cases from JSONL."""
    records = _read_jsonl(path)
    cases: list[RetrievalCase] = []
    query_ids: set[str] = set()
    for line_number, record in records:
        query_id = _required_string(record, "query_id", path, line_number)
        if query_id in query_ids:
            raise ValueError(f"{path}:{line_number}: duplicate query_id {query_id!r}")
        query_ids.add(query_id)
        legacy_value = record.get("relevant_source_ids")
        legacy_relevant = (
            frozenset(
                _string_list(
                    legacy_value, "relevant_source_ids", path, line_number
                )
            )
            if legacy_value is not None
            else frozenset()
        )
        relevance_value = record.get("relevance")
        relevance = (
            _relevance_mapping(relevance_value, path, line_number)
            if relevance_value is not None
            else {source_id: 1.0 for source_id in legacy_relevant}
        )
        positive_ids = frozenset(
            source_id for source_id, value in relevance.items() if value > 0
        )
        if (
            legacy_value is not None
            and relevance_value is not None
            and legacy_relevant != positive_ids
        ):
            raise ValueError(
                f"{path}:{line_number}: relevant_source_ids must match positive "
                "relevance labels"
            )
        answerable_value = record.get("answerable")
        if answerable_value is None:
            if not positive_ids:
                raise ValueError(
                    f"{path}:{line_number}: empty relevance requires answerable=false"
                )
            answerable = True
        elif isinstance(answerable_value, bool):
            answerable = answerable_value
        else:
            raise ValueError(f"{path}:{line_number}: answerable must be a boolean")
        required_value = record.get("required_source_ids")
        required = (
            frozenset(
                _string_list(
                    required_value, "required_source_ids", path, line_number
                )
            )
            if required_value is not None
            else positive_ids
        )
        cases.append(
            RetrievalCase(
                query_id=query_id,
                query=_required_string(record, "query", path, line_number),
                tenant_id=_required_string(record, "tenant_id", path, line_number),
                relevant_source_ids=positive_ids,
                principals=_string_list(
                    record.get("principals", []), "principals", path, line_number
                ),
                relevance=relevance,
                required_source_ids=required,
                answerable=answerable,
            )
        )
    if not cases:
        raise ValueError(f"{path}: cases must contain at least one non-empty JSON line")
    return tuple(cases)


def evaluate_retrieval(
    pipeline: MarkdownBM25Pipeline,
    cases: Sequence[RetrievalCase],
    *,
    k: int,
) -> dict[str, Any]:
    """Evaluate answerable and no-answer cases with explicit denominators."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not cases:
        raise ValueError("at least one retrieval case is required")
    retrieved: dict[str, list[str]] = {}
    relevant: dict[str, set[str]] = {}
    relevance: dict[str, Mapping[str, float]] = {}
    required: dict[str, set[str]] = {}
    case_rows: list[dict[str, Any]] = []
    no_answer_zero_results = 0
    for case in cases:
        all_results = pipeline.retrieve(
            case.query,
            tenant_id=case.tenant_id,
            principals=case.principals,
            top_k=len(pipeline.documents),
        )
        source_ids = list(dict.fromkeys(_source_id(result.document) for result in all_results))[:k]
        retrieved[case.query_id] = source_ids
        found_relevant = case.relevant_source_ids & set(source_ids)
        found_required = case.required_source_ids & set(source_ids)
        judged_nonrelevant = {
            source_id for source_id in source_ids if case.relevance.get(source_id) == 0
        }
        unjudged = {
            source_id for source_id in source_ids if source_id not in case.relevance
        }
        row: dict[str, Any] = {
            "query_id": case.query_id,
            "query": case.query,
            "tenant_id": case.tenant_id,
            "principals": list(case.principals),
            "answerable": case.answerable,
            "retrieved_source_ids": source_ids,
            "relevance": dict(sorted(case.relevance.items())),
            "relevant_source_ids": sorted(case.relevant_source_ids),
            "required_source_ids": sorted(case.required_source_ids),
            "found_relevant_source_ids": sorted(found_relevant),
            "missing_relevant_source_ids": sorted(
                case.relevant_source_ids - found_relevant
            ),
            "found_required_source_ids": sorted(found_required),
            "missing_required_source_ids": sorted(
                case.required_source_ids - found_required
            ),
            "judged_nonrelevant_retrieved_source_ids": sorted(judged_nonrelevant),
            "unjudged_retrieved_source_ids": sorted(unjudged),
            "gold_source_status": _gold_source_status(pipeline, case),
        }
        if case.answerable:
            relevant[case.query_id] = set(case.relevant_source_ids)
            relevance[case.query_id] = case.relevance
            required[case.query_id] = set(case.required_source_ids)
            row["metrics"] = _answerable_case_metrics(
                case.query_id,
                source_ids,
                relevant[case.query_id],
                relevance[case.query_id],
                required[case.query_id],
                k,
            )
        else:
            zero_results = not source_ids
            no_answer_zero_results += int(zero_results)
            row["metrics"] = {
                "zero_results": zero_results,
                "note": "retrieval-layer signal only; it does not prove correct refusal",
            }
        case_rows.append(row)

    answerable_retrieved = {
        query_id: retrieved[query_id] for query_id in relevant
    }
    answerable_count = len(relevant)
    no_answer_count = len(cases) - answerable_count
    answerable_metrics: dict[str, Any] = {"case_count": answerable_count}
    if answerable_count:
        answerable_metrics.update(
            {
                "recall_at_k": recall_at_k(answerable_retrieved, relevant, k=k),
                "mrr_at_k": mean_reciprocal_rank(
                    answerable_retrieved, relevant, k=k
                ),
                "ndcg_at_k": normalized_discounted_cumulative_gain(
                    answerable_retrieved, relevance, k=k
                ),
                "precision_at_k": precision_at_k(
                    answerable_retrieved, relevant, k=k
                ),
                "all_evidence_recall_at_k": all_evidence_recall_at_k(
                    answerable_retrieved, required, k=k
                ),
            }
        )
    no_answer_metrics: dict[str, Any] = {
        "case_count": no_answer_count,
        "zero_result_accuracy": (
            no_answer_zero_results / no_answer_count if no_answer_count else None
        ),
        "note": "zero results are a retrieval-layer signal, not proof of correct refusal",
    }
    report: dict[str, Any] = {
        "k": k,
        "case_count": len(cases),
        "answerable_metrics": answerable_metrics,
        "no_answer_metrics": no_answer_metrics,
        "qrels_warning": (
            "unjudged retrieved sources are not automatically irrelevant when labels are incomplete"
        ),
        "cases": case_rows,
    }
    if answerable_count:
        report.update(
            {
                "recall_at_k": answerable_metrics["recall_at_k"],
                "mrr_at_k": answerable_metrics["mrr_at_k"],
                "ndcg_at_k": answerable_metrics["ndcg_at_k"],
                "precision_at_k": answerable_metrics["precision_at_k"],
                "all_evidence_recall_at_k": answerable_metrics[
                    "all_evidence_recall_at_k"
                ],
                "legacy_metric_scope": "answerable cases only",
            }
        )
    return report


def evaluate_answers(
    pipeline: MarkdownBM25Pipeline,
    cases: Sequence[RetrievalCase],
    answers: Sequence[RecordedAnswer],
) -> dict[str, Any]:
    """Join recorded outputs to cases and independently recheck context visibility."""
    if not cases:
        raise ValueError("at least one retrieval case is required")
    case_by_id = {case.query_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("retrieval cases contain duplicate query ids")
    answer_by_id = {answer.query_id: answer for answer in answers}
    if len(answer_by_id) != len(answers):
        raise ValueError("recorded answers contain duplicate query ids")
    missing = set(case_by_id) - set(answer_by_id)
    extra = set(answer_by_id) - set(case_by_id)
    if missing or extra:
        raise ValueError(
            f"recorded answer query join mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    context_status = {
        query_id: _source_status(
            pipeline,
            tenant_id=case_by_id[query_id].tenant_id,
            principals=case_by_id[query_id].principals,
            source_ids=answer_by_id[query_id].context_source_ids,
        )
        for query_id in case_by_id
    }
    return evaluate_recorded_answers(
        expected_answerable={
            query_id: case.answerable for query_id, case in case_by_id.items()
        },
        answers=answers,
        context_status=context_status,
    )


def evaluate_extractive_baseline(
    pipeline: MarkdownBM25Pipeline,
    cases: Sequence[RetrievalCase],
    *,
    candidate_k: int,
    budget_units: int,
    max_chunks_per_source: int = 2,
    config: ExtractiveAnswerConfig | None = None,
) -> dict[str, Any]:
    """Generate without qrels, then use case labels only for offline evaluation."""
    if not cases:
        raise ValueError("at least one retrieval case is required")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    artifacts: list[ExtractiveAnswerArtifact] = []
    for case in cases:
        # Only the online request fields cross the generator boundary.  In
        # particular, answerable/relevance/required_source_ids are not accepted
        # by generate_extractive_answer.
        results = pipeline.retrieve(
            case.query,
            tenant_id=case.tenant_id,
            principals=case.principals,
            top_k=candidate_k,
        )
        artifacts.append(
            generate_extractive_answer(
                results,
                query_id=case.query_id,
                query=case.query,
                tenant_id=case.tenant_id,
                principals=case.principals,
                cost_fn=utf8_byte_length,
                budget_units=budget_units,
                cost_unit="utf8_bytes",
                max_chunks_per_source=max_chunks_per_source,
                config=config,
            )
        )
    answer_report = evaluate_answers(
        pipeline,
        cases,
        [artifact.recorded_answer for artifact in artifacts],
    )
    return {
        "generator_label_boundary": (
            "qrels and expected answerability are used only after artifact generation"
        ),
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "answer_evaluation": answer_report,
    }


def _answerable_case_metrics(
    query_id: str,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    relevance: Mapping[str, float],
    required_ids: set[str],
    k: int,
) -> dict[str, float]:
    retrieved = {query_id: retrieved_ids}
    return {
        "recall_at_k": recall_at_k(retrieved, {query_id: relevant_ids}, k=k),
        "mrr_at_k": mean_reciprocal_rank(
            retrieved, {query_id: relevant_ids}, k=k
        ),
        "ndcg_at_k": normalized_discounted_cumulative_gain(
            retrieved, {query_id: relevance}, k=k
        ),
        "precision_at_k": precision_at_k(
            retrieved, {query_id: relevant_ids}, k=k
        ),
        "all_evidence_recall_at_k": all_evidence_recall_at_k(
            retrieved, {query_id: required_ids}, k=k
        ),
    }


def _gold_source_status(
    pipeline: MarkdownBM25Pipeline, case: RetrievalCase
) -> dict[str, str]:
    return _source_status(
        pipeline,
        tenant_id=case.tenant_id,
        principals=case.principals,
        source_ids=case.relevant_source_ids,
    )


def _source_status(
    pipeline: MarkdownBM25Pipeline,
    *,
    tenant_id: str,
    principals: Sequence[str],
    source_ids: Iterable[str],
) -> dict[str, str]:
    principal_set = set(principals)
    status: dict[str, str] = {}
    for source_id in sorted(source_ids):
        matching = [
            chunk
            for chunk in pipeline.chunks
            if chunk.source_id == source_id and chunk.tenant_id == tenant_id
        ]
        if not matching:
            status[source_id] = "missing_from_tenant_corpus"
        elif all(chunk.acl and principal_set.isdisjoint(chunk.acl) for chunk in matching):
            status[source_id] = "acl_blocked"
        else:
            status[source_id] = "visible"
    return status


def _chunk_to_document(chunk: SourceChunk) -> Document:
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "source_id": chunk.source_id,
            "source_version": chunk.source_version,
            "heading_path": chunk.heading_path,
            "ordinal": chunk.ordinal,
            "content_hash": chunk.content_hash,
        }
    )
    return Document(
        document_id=chunk.chunk_id,
        text=chunk.text,
        tenant_id=chunk.tenant_id,
        metadata=metadata,
        acl=chunk.acl,
    )


def _source_id(document: Document) -> str:
    source_id = document.metadata.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"document {document.document_id!r} has no source_id metadata")
    return source_id


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value: Any = json.loads(
                line,
                parse_constant=partial(
                    _reject_json_constant, path=path, line_number=line_number
                ),
                parse_float=partial(
                    _finite_json_float, path=path, line_number=line_number
                ),
                object_pairs_hook=partial(
                    _unique_json_object, path=path, line_number=line_number
                ),
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
        records.append((line_number, cast(dict[str, Any], value)))
    return records


def _reject_json_constant(value: str, path: Path, line_number: int) -> None:
    raise ValueError(
        f"{path}:{line_number}: non-finite JSON constant {value!r} is not allowed"
    )


def _finite_json_float(value: str, path: Path, line_number: int) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{path}:{line_number}: non-finite JSON number is not allowed")
    return parsed


def _unique_json_object(
    pairs: list[tuple[str, Any]], path: Path, line_number: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{path}:{line_number}: duplicate JSON key {key!r}")
        result[key] = value
    return result


def _required_string(
    record: Mapping[str, Any], key: str, path: Path, line_number: int
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: {key} must be a non-empty string")
    return value


def _string_list(
    value: Any, key: str, path: Path, line_number: int
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{path}:{line_number}: {key} must be an array of non-empty strings")
    strings = cast(list[str], value)
    if len(strings) != len(set(strings)):
        raise ValueError(f"{path}:{line_number}: {key} contains duplicate values")
    return tuple(strings)


def _relevance_mapping(
    value: Any, path: Path, line_number: int
) -> dict[str, float]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and key.strip() for key in value
    ):
        raise ValueError(
            f"{path}:{line_number}: relevance must be an object keyed by source id"
        )
    labels: dict[str, float] = {}
    for source_id, raw_label in value.items():
        if isinstance(raw_label, bool) or not isinstance(raw_label, (int, float)):
            raise ValueError(
                f"{path}:{line_number}: relevance labels must be finite non-negative numbers"
            )
        label = float(raw_label)
        if not math.isfinite(label) or label < 0:
            raise ValueError(
                f"{path}:{line_number}: relevance labels must be finite non-negative numbers"
            )
        labels[source_id] = label
    return labels


def _result_payload(result: SearchResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "score": result.score,
        "document_id": result.document.document_id,
        "source_id": _source_id(result.document),
        "source_version": result.document.metadata.get("source_version"),
        "heading_path": result.document.metadata.get("heading_path"),
        "text": result.document.text,
    }


def load_recorded_rerank_scores(path: Path) -> tuple[RecordedRerankScore, ...]:
    records: list[RecordedRerankScore] = []
    expected_fields = {
        "query_sha256",
        "document_id",
        "content_sha256",
        "score",
        "scorer_identity",
    }
    for line_number, record in _read_jsonl(path):
        if set(record) != expected_fields:
            missing = sorted(expected_fields - set(record))
            extra = sorted(set(record) - expected_fields)
            raise ValueError(
                f"{path}:{line_number}: rerank score fields mismatch: "
                f"missing={missing}, extra={extra}"
            )
        raw_score = record["score"]
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(raw_score)
        ):
            raise ValueError(f"{path}:{line_number}: score must be a finite number")
        records.append(
            RecordedRerankScore(
                query_sha256=_required_string(
                    record, "query_sha256", path, line_number
                ),
                document_id=_required_string(record, "document_id", path, line_number),
                content_sha256=_required_string(
                    record, "content_sha256", path, line_number
                ),
                score=float(raw_score),
                scorer_identity=_required_string(
                    record, "scorer_identity", path, line_number
                ),
            )
        )
    try:
        return RecordedRerankScorer(records).records
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _run_retrieve(args: argparse.Namespace) -> int:
    pipeline = MarkdownBM25Pipeline(load_corpus(args.corpus), max_chars=args.max_chars)
    results = pipeline.retrieve(
        args.query,
        tenant_id=args.tenant,
        principals=args.principal,
        top_k=args.top_k,
    )
    context = build_citation_context(
        results,
        tenant_id=args.tenant,
        principals=args.principal,
    )
    payload = {
        "query": args.query,
        "tenant_id": args.tenant,
        "principals": args.principal,
        "retrieved": [_result_payload(result) for result in results],
        "context": {
            "rendered": context.rendered,
            "sources": {
                source_id: {
                    "document_id": document.document_id,
                    "source_id": _source_id(document),
                }
                for source_id, document in context.sources.items()
            },
        },
    }
    _print_json(payload)
    return 0


def _run_rerank_recorded(args: argparse.Namespace) -> int:
    if args.candidate_k <= 0:
        raise ValueError("candidate-k must be positive")
    if args.top_k <= 0 or args.top_k > args.candidate_k:
        raise ValueError("top-k must be positive and no greater than candidate-k")
    pipeline = MarkdownBM25Pipeline(load_corpus(args.corpus), max_chars=args.max_chars)
    candidates = pipeline.retrieve(
        args.query,
        tenant_id=args.tenant,
        principals=args.principal,
        top_k=args.candidate_k,
    )
    if not candidates:
        raise ValueError("recorded rerank requires at least one authorized candidate")
    scorer = RecordedRerankScorer(load_recorded_rerank_scores(args.scores))
    report = rerank_authorized_candidates(
        args.query,
        candidates,
        scorer,
        tenant_id=args.tenant,
        principals=args.principal,
        top_k=args.top_k,
        scorer_identity=scorer.scorer_identity,
    )
    _print_json(
        {
            "command": "rerank-recorded",
            "retriever": "bm25",
            "candidate_k": args.candidate_k,
            "candidate_results": [_result_payload(result) for result in candidates],
            "rerank_report": report.to_dict(),
            "evidence_boundary": (
                "Recorded scores prove strict artifact binding and rerank control flow; "
                "they do not prove a learned model ran or improved relevance."
            ),
        }
    )
    return 0


def _select_source(
    corpus: Path, *, tenant_id: str, source_id: str
) -> SourceDocument:
    matches = tuple(
        source
        for source in load_corpus(corpus)
        if source.tenant_id == tenant_id and source.source_id == source_id
    )
    if not matches:
        raise ValueError(
            f"{corpus}: no source matches tenant={tenant_id!r}, source_id={source_id!r}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"{corpus}: tenant/source selection must identify exactly one source"
        )
    return matches[0]


def _run_store_upsert(args: argparse.Namespace) -> int:
    source = _select_source(
        args.corpus,
        tenant_id=args.tenant,
        source_id=args.source_id,
    )
    expected_version = None if args.expect_absent else args.expected_current_version
    if expected_version is not None and not expected_version.strip():
        raise ValueError("expected-current-version must be non-empty")
    with SQLiteChunkStore(
        args.database, busy_timeout_ms=args.busy_timeout_ms
    ) as store:
        plan = store.upsert_source(
            source,
            expected_current_version=expected_version,
            max_chars=args.max_chars,
        )
    _print_json(
        {
            "operation": "upsert",
            "tenant_id": source.tenant_id,
            "source_id": source.source_id,
            "expected_current_version": expected_version,
            "new_version": source.version,
            "plan": {
                "upsert_chunk_ids": [chunk.chunk_id for chunk in plan.upsert],
                "delete_chunk_ids": list(plan.delete_chunk_ids),
                "unchanged_chunk_ids": list(plan.unchanged_chunk_ids),
            },
            "committed": True,
            "scope": {
                "single_sqlite_database_transaction": True,
                "remote_vector_index_updated": False,
                "cross_store_atomicity_proved": False,
            },
        }
    )
    return 0


def _run_store_delete(args: argparse.Namespace) -> int:
    with SQLiteChunkStore(
        args.database, busy_timeout_ms=args.busy_timeout_ms
    ) as store:
        deleted = store.delete_source(
            tenant_id=args.tenant,
            source_id=args.source_id,
            expected_current_version=args.expected_current_version,
        )
    _print_json(
        {
            "operation": "delete",
            "tenant_id": args.tenant,
            "source_id": args.source_id,
            "expected_current_version": args.expected_current_version,
            "deleted_chunk_ids": list(deleted),
            "committed": True,
            "scope": {
                "single_sqlite_database_transaction": True,
                "remote_vector_index_updated": False,
                "cross_store_atomicity_proved": False,
            },
        }
    )
    return 0


def _run_store_retrieve(args: argparse.Namespace) -> int:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    with SQLiteChunkStore(
        args.database, busy_timeout_ms=args.busy_timeout_ms
    ) as store:
        chunks = store.visible_chunks(
            tenant_id=args.tenant,
            principals=args.principal,
        )
    documents = tuple(_chunk_to_document(chunk) for chunk in chunks)
    results = (
        BM25Index(documents).search(
            args.query,
            tenant_id=args.tenant,
            principals=args.principal,
            top_k=args.top_k,
        )
        if documents
        else []
    )
    context = build_citation_context(
        results,
        tenant_id=args.tenant,
        principals=args.principal,
    )
    _print_json(
        {
            "operation": "retrieve",
            "query": args.query,
            "tenant_id": args.tenant,
            "principals": args.principal,
            "authorized_candidate_count": len(documents),
            "retrieved": [_result_payload(result) for result in results],
            "context": {
                "rendered": context.rendered,
                "sources": {
                    short_id: {
                        "document_id": document.document_id,
                        "source_id": _source_id(document),
                    }
                    for short_id, document in context.sources.items()
                },
            },
            "scope": {
                "tenant_filtered_in_sql": True,
                "principal_acl_filtered_before_scoring": True,
                "retriever": "in_memory_bm25_over_authorized_sqlite_chunks",
                "ann_or_embedding_retrieval": False,
            },
        }
    )
    return 0


def _backup_manifest_payload(
    manifest: SQLiteChunkBackupManifest, *, operation: str
) -> dict[str, Any]:
    return {
        "operation": operation,
        "manifest": manifest.to_dict(),
        "scope": {
            "sqlite_quick_check": True,
            "sqlite_foreign_key_check": True,
            "physical_sha256_checked": True,
            "logical_row_fingerprint_checked": True,
            "existing_output_replaced": False,
            "manifest_signature_verified": False,
            "encryption_verified": False,
            "remote_vector_store_included": False,
            "crash_durability_proved": False,
        },
    }


def _run_store_backup(args: argparse.Namespace) -> int:
    manifest = create_sqlite_chunk_backup(
        args.database,
        args.backup,
        args.manifest,
    )
    _print_json(_backup_manifest_payload(manifest, operation="backup"))
    return 0


def _run_store_verify_backup(args: argparse.Namespace) -> int:
    manifest = verify_sqlite_chunk_backup(args.backup, args.manifest)
    payload = _backup_manifest_payload(manifest, operation="verify-backup")
    payload["verified"] = True
    _print_json(payload)
    return 0


def _run_store_restore(args: argparse.Namespace) -> int:
    manifest = restore_sqlite_chunk_backup(
        args.backup,
        args.manifest,
        args.database,
    )
    payload = _backup_manifest_payload(manifest, operation="restore")
    payload["restored"] = True
    _print_json(payload)
    return 0


def _run_pack(args: argparse.Namespace) -> int:
    pipeline = MarkdownBM25Pipeline(load_corpus(args.corpus), max_chars=args.max_chars)
    results = pipeline.retrieve(
        args.query,
        tenant_id=args.tenant,
        principals=args.principal,
        top_k=args.candidate_k,
    )
    packed = pack_citation_context(
        results,
        tenant_id=args.tenant,
        principals=args.principal,
        budget_units=args.budget_bytes,
        cost_fn=utf8_byte_length,
        cost_unit="utf8_bytes",
        max_chunks_per_source=args.max_chunks_per_source,
    )
    _print_json(
        {
            "query": args.query,
            "tenant_id": args.tenant,
            "principals": args.principal,
            "candidate_count": len(results),
            "packing": {
                "cost_unit": packed.cost_unit,
                "budget_units": packed.budget_units,
                "base_cost_units": packed.base_cost_units,
                "used_cost_units": packed.used_cost_units,
                "remaining_units": packed.budget_units - packed.used_cost_units,
                "max_chunks_per_source": packed.max_chunks_per_source,
                "warning": (
                    "UTF-8 serialized bytes are not model tokens; inject the target "
                    "tokenizer/full-prompt cost function in production"
                ),
                "decisions": [
                    {
                        "document_id": decision.document_id,
                        "stable_source_id": decision.stable_source_id,
                        "rank": decision.rank,
                        "selected": decision.selected,
                        "reason": decision.reason.value,
                        "cost_if_selected_units": decision.cost_if_selected_units,
                    }
                    for decision in packed.decisions
                ],
            },
            "context": {
                "rendered": packed.context.rendered,
                "sources": {
                    source_id: {
                        "document_id": document.document_id,
                        "stable_source_id": _source_id(document),
                    }
                    for source_id, document in packed.context.sources.items()
                },
            },
        }
    )
    return 0


def _run_answer_extractive(args: argparse.Namespace) -> int:
    """Run retrieval, authorized packing, and the non-LLM exact-span baseline."""
    pipeline = MarkdownBM25Pipeline(load_corpus(args.corpus), max_chars=args.max_chars)
    results = pipeline.retrieve(
        args.query,
        tenant_id=args.tenant,
        principals=args.principal,
        top_k=args.candidate_k,
    )
    artifact = generate_extractive_answer(
        results,
        query_id=args.query_id,
        query=args.query,
        tenant_id=args.tenant,
        principals=args.principal,
        cost_fn=utf8_byte_length,
        budget_units=args.budget_bytes,
        cost_unit="utf8_bytes",
        max_chunks_per_source=args.max_chunks_per_source,
        config=ExtractiveAnswerConfig(
            min_query_coverage=args.min_query_coverage,
            min_span_matched_tokens=args.min_span_matched_tokens,
            max_answer_spans=args.max_answer_spans,
            max_spans_per_source=args.max_spans_per_source,
        ),
    )
    _print_json(artifact.to_dict())
    return 0


def _run_evaluate_extractive(args: argparse.Namespace) -> int:
    pipeline = MarkdownBM25Pipeline(load_corpus(args.corpus), max_chars=args.max_chars)
    report = evaluate_extractive_baseline(
        pipeline,
        load_cases(args.cases),
        candidate_k=args.candidate_k,
        budget_units=args.budget_bytes,
        max_chunks_per_source=args.max_chunks_per_source,
        config=ExtractiveAnswerConfig(
            min_query_coverage=args.min_query_coverage,
            min_span_matched_tokens=args.min_span_matched_tokens,
            max_answer_spans=args.max_answer_spans,
            max_spans_per_source=args.max_spans_per_source,
        ),
    )
    _print_json(report)
    return 0


def _run_pack_tokenized(args: argparse.Namespace) -> int:
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - exercised in minimal installs
        raise ValueError(
            'pack-tokenized requires the optional "transformers" dependencies'
        ) from error

    if args.max_total_tokens <= 0:
        raise ValueError("max-total-tokens must be positive")
    if args.reserved_output_tokens <= 0:
        raise ValueError("reserved-output-tokens must be positive")
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
    user_prompt_template = args.user_prompt_template_file.read_text(encoding="utf-8")
    transformers_package = cast(Any, transformers)
    tokenizer = transformers_package.AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    chat_template_source = "checkpoint"
    if args.chat_template_path is not None:
        tokenizer.chat_template = args.chat_template_path.read_text(encoding="utf-8")
        chat_template_source = "local_override"
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise ValueError(
            "target tokenizer has no chat_template; provide --chat-template-path"
        )

    latest_token_ids: tuple[int, ...] = ()

    def tokenize_messages(messages: tuple[Mapping[str, str], ...]) -> Iterable[int]:
        nonlocal latest_token_ids
        rendered = tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        if not isinstance(rendered, list):
            raise ValueError("chat tokenizer did not return a token-id list")
        latest_token_ids = tuple(cast(list[int], rendered))
        return latest_token_ids

    cost_fn = make_rag_chat_prompt_cost(
        system_prompt=system_prompt,
        query=args.query,
        user_prompt_template=user_prompt_template,
        tokenize_messages=tokenize_messages,
        reserved_output_tokens=args.reserved_output_tokens,
    )
    pipeline = MarkdownBM25Pipeline(load_corpus(args.corpus), max_chars=args.max_chars)
    results = pipeline.retrieve(
        args.query,
        tenant_id=args.tenant,
        principals=args.principal,
        top_k=args.candidate_k,
    )
    packed = pack_citation_context(
        results,
        tenant_id=args.tenant,
        principals=args.principal,
        budget_units=args.max_total_tokens,
        cost_fn=cost_fn,
        cost_unit=f"tokens:{args.tokenizer}@{args.tokenizer_revision}",
        max_chunks_per_source=args.max_chunks_per_source,
    )
    final_total_cost = cost_fn(packed.context.rendered)
    if final_total_cost != packed.used_cost_units:
        raise ValueError("final prompt tokenization differs from packed cost")
    _print_json(
        {
            "query": args.query,
            "tenant_id": args.tenant,
            "principals": args.principal,
            "candidate_count": len(results),
            "target_tokenizer": {
                "name_or_path": args.tokenizer,
                "revision": args.tokenizer_revision,
                "local_files_only": args.local_files_only,
                "transformers_version": transformers_package.__version__,
                "tokenizer_class": type(tokenizer).__name__,
                "vocabulary_size": len(tokenizer),
                "chat_template_source": chat_template_source,
                "chat_template_sha256": _text_sha256(chat_template),
                "system_prompt_sha256": _text_sha256(system_prompt),
                "user_prompt_template_sha256": _text_sha256(user_prompt_template),
                "add_generation_prompt": True,
            },
            "packing": {
                "cost_unit": packed.cost_unit,
                "max_total_tokens": packed.budget_units,
                "reserved_output_tokens": args.reserved_output_tokens,
                "base_prompt_tokens": (
                    packed.base_cost_units - args.reserved_output_tokens
                ),
                "used_prompt_tokens": (
                    packed.used_cost_units - args.reserved_output_tokens
                ),
                "final_prompt_token_count": len(latest_token_ids),
                "final_prompt_token_ids": list(latest_token_ids),
                "used_total_with_output_reservation": packed.used_cost_units,
                "remaining_total_tokens": (
                    packed.budget_units - packed.used_cost_units
                ),
                "max_chunks_per_source": packed.max_chunks_per_source,
                "decisions": [
                    {
                        "document_id": decision.document_id,
                        "stable_source_id": decision.stable_source_id,
                        "rank": decision.rank,
                        "selected": decision.selected,
                        "reason": decision.reason.value,
                        "prospective_total_with_output_reservation": (
                            decision.cost_if_selected_units
                        ),
                    }
                    for decision in packed.decisions
                ],
            },
            "context": {
                "rendered": packed.context.rendered,
                "sources": {
                    source_id: {
                        "document_id": document.document_id,
                        "stable_source_id": _source_id(document),
                    }
                    for source_id, document in packed.context.sources.items()
                },
            },
            "scope": {
                "complete_chat_prompt_retokenized_per_candidate": True,
                "final_prompt_token_ids_recorded": True,
                "output_token_reservation_included": True,
                "model_context_window_verified": False,
                "generation_quality_or_grounding_verified": False,
                "tokenizer_files_cryptographically_authenticated": False,
            },
            "evidence_boundary": (
                "This report proves packing against the loaded tokenizer/chat template "
                "and supplied max-total/reserved-output settings. It does not prove the "
                "tokenizer matches a deployed model, the configured limit is that model's "
                "usable context window, or generated answers are grounded."
            ),
        }
    )
    return 0


def _text_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _run_evaluate(args: argparse.Namespace) -> int:
    pipeline = MarkdownBM25Pipeline(load_corpus(args.corpus), max_chars=args.max_chars)
    _print_json(evaluate_retrieval(pipeline, load_cases(args.cases), k=args.top_k))
    return 0


def _run_evaluate_answers(args: argparse.Namespace) -> int:
    pipeline = MarkdownBM25Pipeline(load_corpus(args.corpus), max_chars=args.max_chars)
    _print_json(
        evaluate_answers(
            pipeline,
            load_cases(args.cases),
            load_recorded_answers(args.answers),
        )
    )
    return 0


def _run_audit_traces(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    pipeline = MarkdownBM25Pipeline(load_corpus(args.corpus), max_chars=args.max_chars)
    report = audit_rag_generation_traces(
        expected_cases={
            case.query_id: RAGTraceCaseBinding(
                query_sha256=_text_sha256(case.query),
                tenant_id=case.tenant_id,
                principals=case.principals,
            )
            for case in cases
        },
        answers=load_recorded_answers(args.answers),
        traces=load_rag_generation_traces(args.traces),
        documents=pipeline.documents,
    )
    _print_json(report.to_dict())
    return 0 if report.gate_passed else 1


def _run_audit(args: argparse.Namespace) -> int:
    answer = args.answer_file.read_text(encoding="utf-8")
    audit = audit_citations(answer, args.source_id)
    _print_json(
        {
            "syntactically_valid": audit.syntactically_valid,
            "cited_source_ids": audit.cited_source_ids,
            "unknown_source_ids": audit.unknown_source_ids,
            "uncited_paragraphs": audit.uncited_paragraphs,
            "warning": "syntax/coverage only; this does not prove claim-evidence entailment",
        }
    )
    return 0


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="about-llm-rag",
        description="Transparent Markdown -> BM25 -> ACL -> citation RAG baseline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieve = subparsers.add_parser("retrieve", help="retrieve authorized chunks")
    retrieve.add_argument("--corpus", type=Path, required=True)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--tenant", required=True)
    retrieve.add_argument("--principal", action="append", default=[])
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.add_argument("--max-chars", type=int, default=1200)
    retrieve.set_defaults(handler=_run_retrieve)

    rerank_recorded = subparsers.add_parser(
        "rerank-recorded",
        help="rerank authorized BM25 candidates with exact recorded score artifacts",
    )
    rerank_recorded.add_argument("--corpus", type=Path, required=True)
    rerank_recorded.add_argument("--scores", type=Path, required=True)
    rerank_recorded.add_argument("--query", required=True)
    rerank_recorded.add_argument("--tenant", required=True)
    rerank_recorded.add_argument("--principal", action="append", default=[])
    rerank_recorded.add_argument("--candidate-k", type=int, default=20)
    rerank_recorded.add_argument("--top-k", type=int, default=5)
    rerank_recorded.add_argument("--max-chars", type=int, default=1200)
    rerank_recorded.set_defaults(handler=_run_rerank_recorded)

    store_upsert = subparsers.add_parser(
        "store-upsert",
        help="transactionally upsert one selected source into the SQLite chunk store",
    )
    store_upsert.add_argument("--database", type=Path, required=True)
    store_upsert.add_argument("--corpus", type=Path, required=True)
    store_upsert.add_argument("--tenant", required=True)
    store_upsert.add_argument("--source-id", required=True)
    expected = store_upsert.add_mutually_exclusive_group(required=True)
    expected.add_argument("--expect-absent", action="store_true")
    expected.add_argument("--expected-current-version")
    store_upsert.add_argument("--max-chars", type=int, default=1200)
    store_upsert.add_argument("--busy-timeout-ms", type=int, default=5000)
    store_upsert.set_defaults(handler=_run_store_upsert)

    store_delete = subparsers.add_parser(
        "store-delete",
        help="transactionally delete one source at an expected current version",
    )
    store_delete.add_argument("--database", type=Path, required=True)
    store_delete.add_argument("--tenant", required=True)
    store_delete.add_argument("--source-id", required=True)
    store_delete.add_argument("--expected-current-version", required=True)
    store_delete.add_argument("--busy-timeout-ms", type=int, default=5000)
    store_delete.set_defaults(handler=_run_store_delete)

    store_retrieve = subparsers.add_parser(
        "store-retrieve",
        help="retrieve from tenant/ACL-visible chunks persisted in SQLite",
    )
    store_retrieve.add_argument("--database", type=Path, required=True)
    store_retrieve.add_argument("--query", required=True)
    store_retrieve.add_argument("--tenant", required=True)
    store_retrieve.add_argument("--principal", action="append", default=[])
    store_retrieve.add_argument("--top-k", type=int, default=5)
    store_retrieve.add_argument("--busy-timeout-ms", type=int, default=5000)
    store_retrieve.set_defaults(handler=_run_store_retrieve)

    store_backup = subparsers.add_parser(
        "store-backup",
        help="create a no-overwrite SQLite snapshot and strict content manifest",
    )
    store_backup.add_argument("--database", type=Path, required=True)
    store_backup.add_argument("--backup", type=Path, required=True)
    store_backup.add_argument("--manifest", type=Path, required=True)
    store_backup.set_defaults(handler=_run_store_backup)

    store_verify_backup = subparsers.add_parser(
        "store-verify-backup",
        help="verify SQLite integrity plus physical and logical backup identity",
    )
    store_verify_backup.add_argument("--backup", type=Path, required=True)
    store_verify_backup.add_argument("--manifest", type=Path, required=True)
    store_verify_backup.set_defaults(handler=_run_store_verify_backup)

    store_restore = subparsers.add_parser(
        "store-restore",
        help="restore a verified SQLite snapshot to a new, absent database path",
    )
    store_restore.add_argument("--backup", type=Path, required=True)
    store_restore.add_argument("--manifest", type=Path, required=True)
    store_restore.add_argument("--database", type=Path, required=True)
    store_restore.set_defaults(handler=_run_store_restore)

    pack = subparsers.add_parser(
        "pack",
        help="pack authorized results under an explicit UTF-8 byte budget",
    )
    pack.add_argument("--corpus", type=Path, required=True)
    pack.add_argument("--query", required=True)
    pack.add_argument("--tenant", required=True)
    pack.add_argument("--principal", action="append", default=[])
    pack.add_argument("--candidate-k", type=int, default=20)
    pack.add_argument("--budget-bytes", type=int, required=True)
    pack.add_argument("--max-chunks-per-source", type=int, default=2)
    pack.add_argument("--max-chars", type=int, default=1200)
    pack.set_defaults(handler=_run_pack)

    answer_extractive = subparsers.add_parser(
        "answer-extractive",
        help="retrieve and emit a deterministic exact-span answer or abstention",
    )
    answer_extractive.add_argument("--corpus", type=Path, required=True)
    answer_extractive.add_argument("--query-id", required=True)
    answer_extractive.add_argument("--query", required=True)
    answer_extractive.add_argument("--tenant", required=True)
    answer_extractive.add_argument("--principal", action="append", default=[])
    answer_extractive.add_argument("--candidate-k", type=int, default=20)
    answer_extractive.add_argument("--budget-bytes", type=int, default=12000)
    answer_extractive.add_argument("--max-chunks-per-source", type=int, default=2)
    answer_extractive.add_argument("--min-query-coverage", type=float, default=0.55)
    answer_extractive.add_argument("--min-span-matched-tokens", type=int, default=2)
    answer_extractive.add_argument("--max-answer-spans", type=int, default=3)
    answer_extractive.add_argument("--max-spans-per-source", type=int, default=2)
    answer_extractive.add_argument("--max-chars", type=int, default=1200)
    answer_extractive.set_defaults(handler=_run_answer_extractive)

    evaluate_extractive = subparsers.add_parser(
        "evaluate-extractive",
        help="generate exact-span artifacts, then evaluate them against offline labels",
    )
    evaluate_extractive.add_argument("--corpus", type=Path, required=True)
    evaluate_extractive.add_argument("--cases", type=Path, required=True)
    evaluate_extractive.add_argument("--candidate-k", type=int, default=20)
    evaluate_extractive.add_argument("--budget-bytes", type=int, default=12000)
    evaluate_extractive.add_argument("--max-chunks-per-source", type=int, default=2)
    evaluate_extractive.add_argument("--min-query-coverage", type=float, default=0.55)
    evaluate_extractive.add_argument("--min-span-matched-tokens", type=int, default=2)
    evaluate_extractive.add_argument("--max-answer-spans", type=int, default=3)
    evaluate_extractive.add_argument("--max-spans-per-source", type=int, default=2)
    evaluate_extractive.add_argument("--max-chars", type=int, default=1200)
    evaluate_extractive.set_defaults(handler=_run_evaluate_extractive)

    tokenized_pack = subparsers.add_parser(
        "pack-tokenized",
        help="pack authorized results with a target tokenizer and full chat template",
    )
    tokenized_pack.add_argument("--corpus", type=Path, required=True)
    tokenized_pack.add_argument("--query", required=True)
    tokenized_pack.add_argument("--tenant", required=True)
    tokenized_pack.add_argument("--principal", action="append", default=[])
    tokenized_pack.add_argument("--candidate-k", type=int, default=20)
    tokenized_pack.add_argument("--max-total-tokens", type=int, required=True)
    tokenized_pack.add_argument("--reserved-output-tokens", type=int, required=True)
    tokenized_pack.add_argument("--tokenizer", required=True)
    tokenized_pack.add_argument("--tokenizer-revision", required=True)
    tokenized_pack.add_argument("--local-files-only", action="store_true")
    tokenized_pack.add_argument("--chat-template-path", type=Path)
    tokenized_pack.add_argument("--system-prompt-file", type=Path, required=True)
    tokenized_pack.add_argument(
        "--user-prompt-template-file", type=Path, required=True
    )
    tokenized_pack.add_argument("--max-chunks-per-source", type=int, default=2)
    tokenized_pack.add_argument("--max-chars", type=int, default=1200)
    tokenized_pack.set_defaults(handler=_run_pack_tokenized)

    evaluate = subparsers.add_parser("evaluate", help="evaluate source-level retrieval")
    evaluate.add_argument("--corpus", type=Path, required=True)
    evaluate.add_argument("--cases", type=Path, required=True)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--max-chars", type=int, default=1200)
    evaluate.set_defaults(handler=_run_evaluate)

    evaluate_answers_parser = subparsers.add_parser(
        "evaluate-answers",
        help="evaluate recorded answer/abstain/error artifacts and supplied claim labels",
    )
    evaluate_answers_parser.add_argument("--corpus", type=Path, required=True)
    evaluate_answers_parser.add_argument("--cases", type=Path, required=True)
    evaluate_answers_parser.add_argument("--answers", type=Path, required=True)
    evaluate_answers_parser.add_argument("--max-chars", type=int, default=1200)
    evaluate_answers_parser.set_defaults(handler=_run_evaluate_answers)

    audit_traces = subparsers.add_parser(
        "audit-traces",
        help="bind generation traces to cases, current chunks, and recorded answers",
    )
    audit_traces.add_argument("--corpus", type=Path, required=True)
    audit_traces.add_argument("--cases", type=Path, required=True)
    audit_traces.add_argument("--answers", type=Path, required=True)
    audit_traces.add_argument("--traces", type=Path, required=True)
    audit_traces.add_argument("--max-chars", type=int, default=1200)
    audit_traces.set_defaults(handler=_run_audit_traces)

    audit = subparsers.add_parser("audit", help="audit citation ids and paragraph coverage")
    audit.add_argument("--answer-file", type=Path, required=True)
    audit.add_argument("--source-id", action="append", required=True)
    audit.set_defaults(handler=_run_audit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    try:
        return handler(args)
    except (OSError, ValueError, PermissionError, sqlite3.Error) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
