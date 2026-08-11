from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from about_llm.rag import (
    AnswerAction,
    ClaimVerdict,
    Document,
    RAGGenerationTrace,
    RAGTraceCaseBinding,
    RAGTraceSource,
    RecordedAnswer,
    RecordedClaim,
    SearchResult,
    audit_rag_generation_traces,
    build_citation_context,
    load_rag_generation_traces,
    load_recorded_answers,
)


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bundle() -> tuple[
    dict[str, RAGTraceCaseBinding],
    RecordedAnswer,
    RAGGenerationTrace,
    tuple[Document, ...],
]:
    documents = (
        Document(
            document_id="s1:v1:chunk-1",
            text="first authorized chunk",
            tenant_id="tenant-a",
            metadata={"source_id": "s1", "source_version": "v1"},
            acl=("engineering",),
        ),
        Document(
            document_id="s2:v3:chunk-1",
            text="second public chunk",
            tenant_id="tenant-a",
            metadata={"source_id": "s2", "source_version": "v3"},
        ),
    )
    context = build_citation_context(
        [
            SearchResult(document=document, score=0.0, rank=rank, source="fixture")
            for rank, document in enumerate(documents, 1)
        ],
        tenant_id="tenant-a",
        principals=("engineering",),
    )
    answer = RecordedAnswer(
        query_id="q1",
        action=AnswerAction.ANSWER,
        context_source_ids=("s1", "s2"),
        claims=(
            RecordedClaim(
                claim_id="c1",
                text="a fixture claim",
                source_ids=("s1",),
                verdict=ClaimVerdict.SUPPORTED,
                judgment_source="fixture-label",
            ),
        ),
        missing_information=(),
    )
    trace = RAGGenerationTrace(
        trace_id="trace-q1-attempt-1",
        query_id="q1",
        query_sha256=_sha256("fixture query"),
        tenant_id="tenant-a",
        principals=("engineering",),
        sources=tuple(
            RAGTraceSource(
                short_id=short_id,
                stable_source_id=str(document.metadata["source_id"]),
                document_id=document.document_id,
                source_version=str(document.metadata["source_version"]),
                content_sha256=_sha256(document.text),
            )
            for short_id, document in context.sources.items()
        ),
        rendered_context=context.rendered,
        prompt_token_ids=(11, 12, 13),
        tokenizer_revision="tokenizer@example-revision",
        chat_template_sha256=_sha256("chat template"),
        system_prompt_sha256=_sha256("system prompt"),
        user_prompt_template_sha256=_sha256("question: {query}\n{context}"),
        reserved_output_tokens=32,
        generator_revision="model@example-revision",
        raw_output='{"answer":"fixture"}',
        recorded_answer_fingerprint=answer.record_fingerprint,
        metadata={"attempt": 1},
    )
    return (
        {
            "q1": RAGTraceCaseBinding(
                _sha256("fixture query"), "tenant-a", ("engineering",)
            )
        },
        answer,
        trace,
        documents,
    )


def _write_trace(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_trace_round_trip_and_complete_audit_pass(tmp_path: Path) -> None:
    cases, answer, trace, documents = _bundle()
    path = tmp_path / "traces.jsonl"
    _write_trace(path, trace.to_dict())

    loaded = load_rag_generation_traces(path)
    assert loaded == (trace,)
    report = audit_rag_generation_traces(
        expected_cases=cases,
        answers=[answer],
        traces=loaded,
        documents=documents,
    )

    assert report.gate_passed is True
    assert report.findings == ()
    assert report.to_dict()["scope"] == {
        "trace_query_and_security_context_verified": True,
        "trace_answer_canonical_identity_verified": True,
        "trace_context_reconstructed_from_current_chunks": True,
        "raw_output_claim_semantics_verified": False,
        "remote_model_execution_verified": False,
        "artifact_origin_cryptographically_authenticated": False,
        "historical_corpus_availability_proven": False,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_version", r"missing=\['trace_version'\]"),
        ("wrong_version", "expected 'about-llm.rag-generation-trace.v1'"),
        ("unknown", r"unknown=\['unexpected'\]"),
    ],
)
def test_trace_loader_rejects_schema_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _, _, trace, _ = _bundle()
    payload = trace.to_dict()
    if mutation == "missing_version":
        del payload["trace_version"]
    elif mutation == "wrong_version":
        payload["trace_version"] = "about-llm.rag-generation-trace.v999"
    else:
        payload["unexpected"] = True
    path = tmp_path / "bad.jsonl"
    _write_trace(path, payload)

    with pytest.raises(ValueError, match=message, check=lambda error: str(path) in str(error)):
        load_rag_generation_traces(path)


@pytest.mark.parametrize(
    "bad_json",
    [
        '{"trace_version":"v1","trace_version":"v2"}\n',
        '{"trace_version":NaN}\n',
        '{"trace_version":Infinity}\n',
    ],
)
def test_trace_loader_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path: Path, bad_json: str
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(bad_json, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid strict JSON"):
        load_rag_generation_traces(path)


def test_answer_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text('{"query_id":"q1","query_id":"q2"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key 'query_id'"):
        load_recorded_answers(path)


def test_answer_fingerprint_and_raw_output_tampering_are_observable() -> None:
    cases, answer, trace, documents = _bundle()
    altered_answer = replace(answer, missing_information=("tampered",))
    report = audit_rag_generation_traces(
        expected_cases=cases,
        answers=[altered_answer],
        traces=[trace],
        documents=documents,
    )
    altered_output = replace(trace, raw_output=trace.raw_output + " ")

    assert report.finding_counts == {"answer_fingerprint_mismatch": 1}
    assert altered_output.raw_output_sha256 != trace.raw_output_sha256
    assert altered_output.trace_fingerprint != trace.trace_fingerprint


def test_audit_detects_query_binding_and_exact_case_joins() -> None:
    cases, answer, trace, documents = _bundle()
    query_report = audit_rag_generation_traces(
        expected_cases=cases,
        answers=[answer],
        traces=[replace(trace, query_sha256=_sha256("different query"))],
        documents=documents,
    )
    missing_report = audit_rag_generation_traces(
        expected_cases=cases,
        answers=[],
        traces=[],
        documents=documents,
    )

    assert query_report.finding_counts == {"query_fingerprint_mismatch": 1}
    assert missing_report.finding_counts == {"missing_answer": 1, "missing_trace": 1}


@pytest.mark.parametrize(
    ("trace_mutator", "document_mutator", "expected_code"),
    [
        (
            lambda trace: replace(
                trace,
                sources=(
                    replace(trace.sources[0], document_id="unknown-chunk"),
                    trace.sources[1],
                ),
            ),
            lambda documents: documents,
            "unknown_document",
        ),
        (
            lambda trace: replace(
                trace,
                sources=(
                    replace(trace.sources[0], source_version="v2"),
                    trace.sources[1],
                ),
            ),
            lambda documents: documents,
            "source_version_mismatch",
        ),
        (
            lambda trace: trace,
            lambda documents: (
                replace(documents[0], text="mutated current chunk bytes"),
                documents[1],
            ),
            "source_content_mismatch",
        ),
    ],
)
def test_audit_detects_chunk_identity_version_and_content_changes(
    trace_mutator: object,
    document_mutator: object,
    expected_code: str,
) -> None:
    cases, answer, trace, documents = _bundle()
    mutated_trace = trace_mutator(trace)  # type: ignore[operator]
    mutated_documents = document_mutator(documents)  # type: ignore[operator]

    report = audit_rag_generation_traces(
        expected_cases=cases,
        answers=[answer],
        traces=[mutated_trace],
        documents=mutated_documents,
    )

    assert expected_code in report.finding_counts
    assert report.gate_passed is False


def test_audit_detects_security_context_context_rendering_and_source_order() -> None:
    cases, answer, trace, documents = _bundle()
    bad_security_trace = replace(trace, tenant_id="tenant-b", principals=())
    security_report = audit_rag_generation_traces(
        expected_cases=cases,
        answers=[answer],
        traces=[bad_security_trace],
        documents=documents,
    )
    bad_context_trace = replace(trace, rendered_context=trace.rendered_context + "\n")
    context_report = audit_rag_generation_traces(
        expected_cases=cases,
        answers=[answer],
        traces=[bad_context_trace],
        documents=documents,
    )
    reversed_sources = replace(
        trace,
        sources=(
            replace(trace.sources[1], short_id="S1"),
            replace(trace.sources[0], short_id="S2"),
        ),
    )
    order_report = audit_rag_generation_traces(
        expected_cases=cases,
        answers=[answer],
        traces=[reversed_sources],
        documents=documents,
    )

    assert {"tenant_mismatch", "principals_mismatch", "document_tenant_mismatch"} <= set(
        security_report.finding_counts
    )
    assert "source_not_authorized" in security_report.finding_counts
    assert context_report.finding_counts == {"rendered_context_mismatch": 1}
    assert "context_source_order_mismatch" in order_report.finding_counts
    assert "rendered_context_mismatch" in order_report.finding_counts


def test_multiple_chunks_from_one_stable_source_are_supported() -> None:
    cases, answer, trace, documents = _bundle()
    extra = Document(
        document_id="s1:v1:chunk-2",
        text="another chunk from the first stable source",
        tenant_id="tenant-a",
        metadata={"source_id": "s1", "source_version": "v1"},
        acl=("engineering",),
    )
    reordered_documents = (documents[0], extra, documents[1])
    context = build_citation_context(
        [
            SearchResult(document=document, score=0.0, rank=rank, source="fixture")
            for rank, document in enumerate(reordered_documents, 1)
        ],
        tenant_id="tenant-a",
        principals=("engineering",),
    )
    multi_chunk_trace = replace(
        trace,
        sources=tuple(
            RAGTraceSource(
                short_id=short_id,
                stable_source_id=str(document.metadata["source_id"]),
                document_id=document.document_id,
                source_version=str(document.metadata["source_version"]),
                content_sha256=_sha256(document.text),
            )
            for short_id, document in context.sources.items()
        ),
        rendered_context=context.rendered,
    )

    report = audit_rag_generation_traces(
        expected_cases=cases,
        answers=[answer],
        traces=[multi_chunk_trace],
        documents=reordered_documents,
    )
    assert report.gate_passed is True
