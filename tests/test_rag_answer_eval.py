from __future__ import annotations

from pathlib import Path

import pytest

from about_llm.rag import (
    AnswerAction,
    ClaimVerdict,
    RecordedAnswer,
    RecordedClaim,
    evaluate_recorded_answers,
    load_recorded_answers,
)
from about_llm.rag.cli import (
    MarkdownBM25Pipeline,
    RetrievalCase,
    evaluate_answers,
    load_cases,
    load_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "rag-foundations"
CORPUS = PROJECT / "sample_corpus.jsonl"
CASES = PROJECT / "sample_eval.jsonl"
ANSWERS = PROJECT / "sample_answers.jsonl"
pytestmark = [pytest.mark.formula, pytest.mark.security]


def claim(
    claim_id: str,
    *,
    sources: tuple[str, ...] = ("rag-evaluation",),
    verdict: ClaimVerdict = ClaimVerdict.SUPPORTED,
    judgment_source: str | None = "test-label",
) -> RecordedClaim:
    return RecordedClaim(
        claim_id=claim_id,
        text=f"claim {claim_id}",
        source_ids=sources,
        verdict=verdict,
        judgment_source=judgment_source,
    )


def test_sample_recorded_answers_keep_metric_denominators_explicit() -> None:
    pipeline = MarkdownBM25Pipeline(load_corpus(CORPUS))
    report = evaluate_answers(
        pipeline,
        load_cases(CASES),
        load_recorded_answers(ANSWERS),
    )

    assert report["case_count"] == 5
    assert report["answerable_case_count"] == 3
    assert report["no_answer_case_count"] == 2
    assert report["answered_case_count"] == 3
    assert report["coverage"] == pytest.approx(0.6)
    assert report["action_accuracy"] == 1.0
    assert report["answerable_action_accuracy"] == 1.0
    assert report["no_answer_abstention_action_accuracy"] == 1.0
    assert report["grounded_answer_pass_rate"] == 1.0
    assert report["recorded_gate_pass_rate"] == 1.0
    assert report["claim_count"] == 4
    assert report["citation_coverage"] == 1.0
    assert report["citation_validity"] == 1.0
    assert report["claim_judgment_coverage"] == 1.0
    assert report["supported_claim_rate"] == 1.0
    assert "supplied labels" in report["scope_warning"]

    rows = {row["query_id"]: row for row in report["cases"]}
    assert rows["topical-no-answer"]["actual_action"] == "abstain"
    assert rows["topical-no-answer"]["context_source_status"] == {
        "rag-evaluation": "visible",
        "rag-security": "visible",
    }


def test_supported_label_does_not_override_acl_or_missing_citation() -> None:
    pipeline = MarkdownBM25Pipeline(load_corpus(CORPUS))
    case = RetrievalCase(
        query_id="q",
        query="SFT",
        tenant_id="tenant-a",
        relevant_source_ids=frozenset({"finetuning-basics"}),
    )
    answer = RecordedAnswer(
        query_id="q",
        action=AnswerAction.ANSWER,
        context_source_ids=("finetuning-basics",),
        claims=(
            claim("hidden", sources=("finetuning-basics",)),
            claim(
                "uncited",
                sources=(),
                verdict=ClaimVerdict.UNJUDGED,
                judgment_source=None,
            ),
        ),
        missing_information=(),
    )

    report = evaluate_answers(pipeline, [case], [answer])
    row = report["cases"][0]
    assert row["context_source_status"] == {"finetuning-basics": "acl_blocked"}
    assert row["grounded_answer_pass"] is False
    assert row["recorded_gate_pass"] is False
    assert "context_not_authorized_or_missing" in row["diagnostics"]
    assert "uncited_claim:uncited" in row["diagnostics"]
    assert report["citation_coverage"] == pytest.approx(0.5)
    assert report["citation_validity"] == 0.0
    assert report["claim_judgment_coverage"] == pytest.approx(0.5)
    assert report["supported_claim_rate"] == 1.0


def test_grounded_answer_can_still_be_wrong_action_for_no_answer_case() -> None:
    answer = RecordedAnswer(
        query_id="q",
        action=AnswerAction.ANSWER,
        context_source_ids=("s",),
        claims=(claim("c", sources=("s",)),),
        missing_information=(),
    )
    report = evaluate_recorded_answers(
        expected_answerable={"q": False},
        answers=[answer],
        context_status={"q": {"s": "visible"}},
    )

    row = report["cases"][0]
    assert row["grounded_answer_pass"] is True
    assert row["action_correct"] is False
    assert row["recorded_gate_pass"] is False
    assert report["no_answer_abstention_action_accuracy"] == 0.0


def test_contradicted_insufficient_and_error_are_first_class_failures() -> None:
    bad_answer = RecordedAnswer(
        query_id="answerable",
        action=AnswerAction.ANSWER,
        context_source_ids=("s",),
        claims=(
            claim("contra", sources=("s",), verdict=ClaimVerdict.CONTRADICTED),
            claim("weak", sources=("s",), verdict=ClaimVerdict.INSUFFICIENT),
        ),
        missing_information=(),
    )
    error = RecordedAnswer(
        query_id="no-answer",
        action=AnswerAction.ERROR,
        context_source_ids=(),
        claims=(),
        missing_information=(),
        error_type="timeout",
    )
    report = evaluate_recorded_answers(
        expected_answerable={"answerable": True, "no-answer": False},
        answers=[bad_answer, error],
        context_status={"answerable": {"s": "visible"}, "no-answer": {}},
    )

    assert report["error_case_count"] == 1
    assert report["coverage"] == pytest.approx(0.5)
    assert report["supported_claim_rate"] == 0.0
    assert report["claim_verdict_counts"] == {
        "supported": 0,
        "contradicted": 1,
        "insufficient": 1,
        "unjudged": 0,
    }
    assert report["recorded_gate_pass_rate"] == 0.0


def test_context_status_never_reveals_cross_tenant_source_existence() -> None:
    pipeline = MarkdownBM25Pipeline(load_corpus(CORPUS))
    case = RetrievalCase(
        query_id="q",
        query="secret",
        tenant_id="tenant-a",
        relevant_source_ids=frozenset(),
        answerable=False,
    )
    answer = RecordedAnswer(
        query_id="q",
        action=AnswerAction.ABSTAIN,
        context_source_ids=("tenant-b-secret",),
        claims=(),
        missing_information=("no authorized evidence",),
    )

    report = evaluate_answers(pipeline, [case], [answer])
    row = report["cases"][0]
    assert row["context_source_status"] == {
        "tenant-b-secret": "missing_from_tenant_corpus"
    }
    assert row["recorded_gate_pass"] is False


def test_record_types_enforce_judgment_and_terminal_action_contracts() -> None:
    with pytest.raises(ValueError, match="judgment_source"):
        claim("missing-provenance", judgment_source=None)
    with pytest.raises(ValueError, match="unjudged claim"):
        claim(
            "false-provenance",
            verdict=ClaimVerdict.UNJUDGED,
            judgment_source="not-allowed",
        )
    with pytest.raises(ValueError, match="at least one atomic claim"):
        RecordedAnswer("q", AnswerAction.ANSWER, (), (), ())
    with pytest.raises(ValueError, match="missing_information"):
        RecordedAnswer("q", AnswerAction.ABSTAIN, (), (), ())
    with pytest.raises(ValueError, match="error_type"):
        RecordedAnswer("q", AnswerAction.ERROR, (), (), ())


def test_loader_rejects_unknown_fields_and_duplicate_query_ids(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(
        '{"query_id":"q","action":"abstain","context_source_ids":[],'
        '"claims":[],"missing_information":["missing"],"typo":true}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        load_recorded_answers(unknown)

    duplicate = tmp_path / "duplicate.jsonl"
    row = (
        '{"query_id":"q","action":"abstain","context_source_ids":[],'
        '"claims":[],"missing_information":["missing"]}\n'
    )
    duplicate.write_text(row + row, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate query_id"):
        load_recorded_answers(duplicate)


def test_evaluator_requires_an_exact_case_output_join() -> None:
    answer = RecordedAnswer(
        query_id="extra",
        action=AnswerAction.ABSTAIN,
        context_source_ids=(),
        claims=(),
        missing_information=("missing",),
    )
    with pytest.raises(ValueError, match="query join mismatch"):
        evaluate_recorded_answers(
            expected_answerable={"expected": False},
            answers=[answer],
            context_status={"extra": {}},
        )
