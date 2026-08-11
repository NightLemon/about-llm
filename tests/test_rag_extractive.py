from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from about_llm.rag.answer_eval import AnswerAction
from about_llm.rag.citations import build_citation_context
from about_llm.rag.cli import (
    MarkdownBM25Pipeline,
    evaluate_answers,
    evaluate_extractive_baseline,
    load_cases,
    load_corpus,
    main,
)
from about_llm.rag.context_packing import PackingReason, utf8_byte_length
from about_llm.rag.extractive import ExtractiveAnswerArtifact, generate_extractive_answer
from about_llm.rag.models import Document, SearchResult

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "rag-foundations"
CORPUS = PROJECT / "sample_corpus.jsonl"
CASES = PROJECT / "sample_eval.jsonl"


def _fixture_artifacts() -> dict[str, ExtractiveAnswerArtifact]:
    pipeline = MarkdownBM25Pipeline(load_corpus(CORPUS))
    artifacts = {}
    for case in load_cases(CASES):
        results = pipeline.retrieve(
            case.query,
            tenant_id=case.tenant_id,
            principals=case.principals,
            top_k=20,
        )
        artifacts[case.query_id] = generate_extractive_answer(
            results,
            query_id=case.query_id,
            query=case.query,
            tenant_id=case.tenant_id,
            principals=case.principals,
            cost_fn=utf8_byte_length,
            budget_units=12000,
            cost_unit="utf8_bytes",
        )
    return artifacts


def test_fixture_answers_and_abstentions_do_not_consult_qrels() -> None:
    artifacts = _fixture_artifacts()

    assert artifacts["acl-before-ranking"].action is AnswerAction.ANSWER
    assert artifacts["retrieval-metrics"].action is AnswerAction.ANSWER
    multi = artifacts["metrics-and-entailment"]
    assert multi.action is AnswerAction.ANSWER
    assert {span.stable_source_id for span in multi.proposed_spans} == {
        "rag-security",
        "rag-evaluation",
    }
    assert artifacts["unrelated-no-answer"].action is AnswerAction.ABSTAIN
    topical = artifacts["topical-no-answer"]
    assert topical.action is AnswerAction.ABSTAIN
    assert topical.coverage == pytest.approx(2 / 9)
    assert topical.answer_text == "证据不足，无法基于已授权知识库回答。"  # noqa: RUF001


def test_every_proposed_span_is_an_exact_authorized_packed_substring() -> None:
    for artifact in _fixture_artifacts().values():
        for span in artifact.proposed_spans:
            document = artifact.packed_context.context.sources[span.short_source_id]
            assert document.tenant_id == artifact.tenant_id
            assert not document.acl or set(document.acl) & set(artifact.principals)
            assert document.text[span.start_char : span.end_char] == span.text
            assert f"{span.text} [{span.short_source_id}]" in (
                artifact.answer_text
                if artifact.action is AnswerAction.ANSWER
                else f"{span.text} [{span.short_source_id}]"
            )


def test_generated_records_pass_existing_answer_gate_on_fixture() -> None:
    pipeline = MarkdownBM25Pipeline(load_corpus(CORPUS))
    cases = load_cases(CASES)
    artifacts = _fixture_artifacts()
    report = evaluate_answers(
        pipeline,
        cases,
        [artifacts[case.query_id].recorded_answer for case in cases],
    )

    assert report["action_accuracy"] == 1.0
    assert report["grounded_answer_pass_rate"] == 1.0
    assert report["recorded_gate_pass_rate"] == 1.0
    assert report["claim_judgment_coverage"] == 1.0
    assert "supplied labels" in report["scope_warning"]


def test_qrel_changes_cannot_change_generated_artifact() -> None:
    pipeline = MarkdownBM25Pipeline(load_corpus(CORPUS))
    original = load_cases(CASES)[0]
    relabeled = replace(
        original,
        relevant_source_ids=frozenset(),
        relevance={},
        required_source_ids=frozenset(),
        answerable=False,
    )

    expected_answer_report = evaluate_extractive_baseline(
        pipeline,
        [original],
        candidate_k=20,
        budget_units=12000,
    )
    expected_abstain_report = evaluate_extractive_baseline(
        pipeline,
        [relabeled],
        candidate_k=20,
        budget_units=12000,
    )

    first = expected_answer_report["artifacts"][0]
    second = expected_abstain_report["artifacts"][0]
    assert first == second
    assert expected_answer_report["answer_evaluation"]["action_accuracy"] == 1.0
    assert expected_abstain_report["answer_evaluation"]["action_accuracy"] == 0.0


def test_packing_rejects_cross_tenant_and_acl_blocked_candidates() -> None:
    cross_tenant = SearchResult(
        document=Document("d1", "alpha beta", "tenant-b"),
        score=1.0,
        rank=1,
        source="test",
    )
    with pytest.raises(PermissionError, match="belongs to tenant"):
        generate_extractive_answer(
            [cross_tenant],
            query_id="q",
            query="alpha beta",
            tenant_id="tenant-a",
            cost_fn=utf8_byte_length,
            budget_units=1000,
            cost_unit="utf8_bytes",
        )

    acl_blocked = SearchResult(
        document=Document("d2", "alpha beta", "tenant-a", acl=("engineering",)),
        score=1.0,
        rank=1,
        source="test",
    )
    with pytest.raises(PermissionError, match="not visible"):
        generate_extractive_answer(
            [acl_blocked],
            query_id="q",
            query="alpha beta",
            tenant_id="tenant-a",
            cost_fn=utf8_byte_length,
            budget_units=1000,
            cost_unit="utf8_bytes",
        )


def test_budget_dropped_document_cannot_supply_an_extractive_span() -> None:
    first = SearchResult(
        document=Document(
            "d1",
            "alpha beta gamma.",
            "tenant-a",
            metadata={"source_id": "source-1", "source_version": "v1"},
        ),
        score=2.0,
        rank=1,
        source="test",
    )
    second = SearchResult(
        document=Document(
            "d2",
            "delta epsilon.",
            "tenant-a",
            metadata={"source_id": "source-2", "source_version": "v1"},
        ),
        score=1.0,
        rank=2,
        source="test",
    )
    one_source_context = build_citation_context([first], tenant_id="tenant-a")
    artifact = generate_extractive_answer(
        [first, second],
        query_id="q",
        query="alpha beta gamma delta",
        tenant_id="tenant-a",
        cost_fn=utf8_byte_length,
        budget_units=utf8_byte_length(one_source_context.rendered),
        cost_unit="utf8_bytes",
    )

    assert artifact.action is AnswerAction.ANSWER
    assert artifact.packed_context.decisions[1].reason is PackingReason.BUDGET
    assert {span.document_id for span in artifact.proposed_spans} == {"d1"}
    assert "d2" not in artifact.answer_text


def test_artifact_validation_rejects_unknown_or_tampered_span() -> None:
    artifact = _fixture_artifacts()["acl-before-ranking"]
    span = artifact.proposed_spans[0]

    with pytest.raises(ValueError, match="outside packed context"):
        replace(
            artifact,
            proposed_spans=(replace(span, short_source_id="S999"),),
        )
    with pytest.raises(ValueError, match="not an exact source substring"):
        replace(
            artifact,
            proposed_spans=(replace(span, text=span.text + "伪造"),),
        )


def test_artifact_and_cli_output_are_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    first = _fixture_artifacts()["metrics-and-entailment"]
    second = _fixture_artifacts()["metrics-and-entailment"]
    assert first.to_dict() == second.to_dict()
    assert first.artifact_fingerprint == second.artifact_fingerprint

    exit_code = main(
        [
            "answer-extractive",
            "--corpus",
            str(CORPUS),
            "--query-id",
            "acl-before-ranking",
            "--query",
            "RAG 为什么要先做 ACL 权限过滤",
            "--tenant",
            "tenant-a",
            "--principal",
            "engineering",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"artifact_version": "about-llm.rag-extractive-answer.v1"' in output
    assert '"action": "answer"' in output
    assert "deterministic-exact-source-span-v1" in output
