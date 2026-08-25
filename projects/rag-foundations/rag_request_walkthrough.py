"""Walk one answerable and one no-answer request through the RAG controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from about_llm.rag import (
    RecordedRerankScorer,
    SearchResult,
    audit_citations,
    generate_extractive_answer,
    rerank_authorized_candidates,
    utf8_byte_length,
)
from about_llm.rag.cli import (
    MarkdownBM25Pipeline,
    load_corpus,
    load_recorded_rerank_scores,
)

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects" / "rag-foundations"
CORPUS = PROJECT / "sample_corpus.jsonl"
RERANK_SCORES = PROJECT / "reranker-scores.example.jsonl"
TENANT_ID = "tenant-a"
PRINCIPALS = ("engineering",)


class _BM25ScorePassthrough:
    """Keep BM25 order while exercising the authorization-first rerank boundary."""

    def score(
        self,
        query: str,
        candidates: tuple[SearchResult, ...],
    ) -> tuple[float, ...]:
        del query
        return tuple(candidate.score for candidate in candidates)


def _result_rows(results: tuple[SearchResult, ...] | list[SearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "rank": result.rank,
            "document_id": result.document.document_id,
            "stable_source_id": result.document.metadata["source_id"],
            "score": result.score,
            "source": result.source,
        }
        for result in results
    ]


def _packing_rows(artifact: Any) -> list[dict[str, Any]]:
    return [
        {
            "document_id": decision.document_id,
            "stable_source_id": decision.stable_source_id,
            "selected": decision.selected,
            "reason": decision.reason.value,
            "cost_if_selected_units": decision.cost_if_selected_units,
        }
        for decision in artifact.packed_context.decisions
    ]


def _request_row(
    *,
    query_id: str,
    query: str,
    candidates: list[SearchResult],
    reranked: tuple[SearchResult, ...],
    scorer_identity: str,
) -> dict[str, Any]:
    artifact = generate_extractive_answer(
        reranked,
        query_id=query_id,
        query=query,
        tenant_id=TENANT_ID,
        principals=PRINCIPALS,
        cost_fn=utf8_byte_length,
        budget_units=12_000,
        cost_unit="utf8_bytes",
    )
    source_ids = tuple(artifact.packed_context.context.sources)
    citation_audit = audit_citations(artifact.answer_text, source_ids)
    if artifact.action.value == "answer":
        citation_required = True
        citation_passed = bool(citation_audit.cited_source_ids) and (
            citation_audit.syntactically_valid
        )
        citation_syntax_status = "passed" if citation_passed else "failed"
        entailment_status = "not_checked"
        final_action = "answer" if citation_passed else "reject"
        final_reason = (
            "exact_span_and_citation_syntax_passed"
            if citation_passed
            else "citation_syntax_failed"
        )
    elif artifact.action.value == "abstain":
        citation_required = False
        citation_syntax_status = "not_applicable"
        entailment_status = "not_applicable"
        final_action = "abstain"
        final_reason = "insufficient_lexical_evidence"
    else:
        raise AssertionError(f"unexpected extractive action {artifact.action.value!r}")
    return {
        "query_id": query_id,
        "query": query,
        "trusted_security_context": {
            "tenant_id": TENANT_ID,
            "principals": list(PRINCIPALS),
        },
        "retrieval": {
            "candidate_count": len(candidates),
            "candidates": _result_rows(candidates),
        },
        "rerank": {
            "scorer_identity": scorer_identity,
            "results": _result_rows(reranked),
        },
        "packing": {
            "cost_unit": artifact.packed_context.cost_unit,
            "used_cost_units": artifact.packed_context.used_cost_units,
            "source_map": {
                short_id: document.metadata["source_id"]
                for short_id, document in artifact.packed_context.context.sources.items()
            },
            "decisions": _packing_rows(artifact),
        },
        "answer": {
            "action": artifact.action.value,
            "text": artifact.answer_text,
            "meaningful_query_tokens": list(artifact.meaningful_query_tokens),
            "covered_query_tokens": list(artifact.covered_query_tokens),
            "coverage": artifact.coverage,
        },
        "citation": {
            "required_for_action": citation_required,
            "known_source_ids": list(source_ids),
            "cited_source_ids": list(citation_audit.cited_source_ids),
            "syntax_status": citation_syntax_status,
            "semantic_entailment_status": entailment_status,
        },
        "final": {
            "action": final_action,
            "reason": final_reason,
        },
    }


def build_walkthrough() -> dict[str, Any]:
    pipeline = MarkdownBM25Pipeline(load_corpus(CORPUS))

    answerable_query = "RAG 为什么要先做 ACL 权限过滤"
    answerable_candidates = pipeline.retrieve(
        answerable_query,
        tenant_id=TENANT_ID,
        principals=PRINCIPALS,
        top_k=3,
    )
    recorded_scorer = RecordedRerankScorer(
        load_recorded_rerank_scores(RERANK_SCORES)
    )
    answerable_rerank = rerank_authorized_candidates(
        answerable_query,
        answerable_candidates,
        tenant_id=TENANT_ID,
        principals=PRINCIPALS,
        scorer=recorded_scorer,
        scorer_identity=recorded_scorer.scorer_identity,
        top_k=2,
    )

    no_answer_query = "引用的 Kubernetes 灾难恢复步骤是什么"
    no_answer_candidates = pipeline.retrieve(
        no_answer_query,
        tenant_id=TENANT_ID,
        principals=PRINCIPALS,
        top_k=3,
    )
    passthrough_identity = "bm25-score-passthrough@teaching-v1"
    no_answer_rerank = rerank_authorized_candidates(
        no_answer_query,
        no_answer_candidates,
        tenant_id=TENANT_ID,
        principals=PRINCIPALS,
        scorer=_BM25ScorePassthrough(),
        scorer_identity=passthrough_identity,
        top_k=3,
    )

    return {
        "walkthrough_version": "about-llm.rag-request-walkthrough.v2",
        "requests": [
            _request_row(
                query_id="request-a-answerable",
                query=answerable_query,
                candidates=answerable_candidates,
                reranked=answerable_rerank.results,
                scorer_identity=recorded_scorer.scorer_identity,
            ),
            _request_row(
                query_id="request-b-no-answer",
                query=no_answer_query,
                candidates=no_answer_candidates,
                reranked=no_answer_rerank.results,
                scorer_identity=passthrough_identity,
            ),
        ],
        "scope": {
            "learned_reranker_executed": False,
            "llm_executed": False,
            "target_tokenizer_used": False,
            "authorization_rechecked_before_reranker": True,
            "exact_source_span_verified_for_answers": True,
            "semantic_entailment_or_source_truth_verified": False,
            "production_quality_latency_or_safety_verified": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rendered = json.dumps(build_walkthrough(), ensure_ascii=False, indent=2) + "\n"
    sys.stdout.buffer.write(rendered.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
