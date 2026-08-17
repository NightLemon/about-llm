from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)
from about_llm.llmops import artifact_fingerprint
from about_llm.rag.citations import CitationContext, build_citation_context
from about_llm.rag.generation_policy import (
    RAG_PUBLICATION_POLICY_VERSION,
    RAG_PUBLICATION_REPLAY_EVIDENCE_BOUNDARY,
    RAG_PUBLICATION_REPLAY_REPORT_VERSION,
    PublicationAction,
    PublicationStage,
    RAGPublicationPolicy,
    build_publication_policy_replay_report,
    evaluate_post_generation,
    evaluate_pre_generation,
    guard_rag_generation,
    verify_publication_policy_replay_report,
)
from about_llm.rag.models import Document, SearchResult
from about_llm.rag.transformers_control import (
    load_rag_transformers_control_spec,
    verify_recorded_rag_transformers_report,
)

ROOT = Path(__file__).resolve().parents[1]
RAG_DIRECTORY = ROOT / "projects" / "rag-foundations"
MANIFEST = RAG_DIRECTORY / "qwen2.5-0.5b-rag.control.json"
RECORDED_REPORT = RAG_DIRECTORY / "qwen2.5-0.5b-rag.recorded-report.json"
FROZEN_REPLAY_REPORT = (
    RAG_DIRECTORY / "qwen2.5-0.5b-rag.publication-policy-replay.json"
)
REPLAY_SCRIPT = RAG_DIRECTORY / "replay_qwen_rag_publication_policy.py"
CHECKPOINT_MANIFEST = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)


def _context() -> CitationContext:
    document = Document(
        document_id="acl-order-v1",
        text="检索必须先执行租户和 ACL 过滤，再对可见候选打分。",  # noqa: RUF001
        tenant_id="tenant-a",
        acl=("engineering",),
    )
    return build_citation_context(
        [SearchResult(document=document, score=2.0, rank=1, source="fixture")],
        tenant_id="tenant-a",
        principals=("engineering",),
    )


def _verified_qwen_source() -> tuple[Any, dict[str, Any]]:
    spec = load_rag_transformers_control_spec(MANIFEST)
    checkpoint = load_checkpoint_control_spec(CHECKPOINT_MANIFEST)
    report = verify_recorded_rag_transformers_report(
        RECORDED_REPORT,
        spec=spec,
        checkpoint_spec=checkpoint,
    )
    return spec, dict(report)


def _rehash(value: dict[str, Any], fingerprint_field: str) -> None:
    unsigned = dict(value)
    unsigned.pop(fingerprint_field, None)
    value[fingerprint_field] = "sha256:" + artifact_fingerprint(unsigned)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_empty_authorized_context_short_circuits_without_calling_generator() -> None:
    context = build_citation_context([], tenant_id="tenant-a")
    calls = 0

    def generator(_: str) -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("generator must not be called without authorized evidence")

    decision = guard_rag_generation(context, generator)

    assert calls == 0
    assert decision.stage is PublicationStage.PRE_GENERATION
    assert decision.action is PublicationAction.ABSTAIN
    assert decision.reason_code == "no_authorized_evidence"
    assert decision.model_call_allowed is False
    assert decision.generated_output_observed is False
    assert decision.raw_output is None
    assert decision.response_text == "无法根据提供的证据回答。"
    assert decision.semantic_entailment_verified is False


def test_valid_local_citation_syntax_publishes_raw_output() -> None:
    context = _context()
    raw_output = "必须先过滤权限，再对可见候选打分。[S1]"  # noqa: RUF001

    decision = guard_rag_generation(context, lambda rendered: raw_output)

    assert '<source id="S1"' in context.rendered
    assert decision.stage is PublicationStage.POST_GENERATION
    assert decision.action is PublicationAction.PUBLISH
    assert decision.reason_code == "citation_syntax_passed"
    assert decision.response_text == raw_output
    assert decision.raw_output == raw_output
    assert decision.cited_source_ids == ("S1",)
    assert decision.unknown_source_ids == ()
    assert decision.uncited_paragraphs == ()
    assert decision.citation_syntax_passed is True
    assert decision.semantic_entailment_verified is False


@pytest.mark.parametrize(
    ("raw_output", "reason_code", "unknown", "uncited"),
    [
        ("答案没有引用。", "missing_citation", (), ("答案没有引用。",)),
        ("引用不存在的来源。[S9]", "unknown_citation", ("S9",), ()),
        (
            "第一段有本地引用。[S1]\n\n第二段没有引用。",
            "uncited_paragraph",
            (),
            ("第二段没有引用。",),
        ),
    ],
)
def test_failed_citation_syntax_rejects_raw_output(
    raw_output: str,
    reason_code: str,
    unknown: tuple[str, ...],
    uncited: tuple[str, ...],
) -> None:
    decision = evaluate_post_generation(_context(), raw_output)

    assert decision.action is PublicationAction.REJECT
    assert decision.reason_code == reason_code
    assert decision.raw_output == raw_output
    assert decision.response_text == "无法生成满足引用要求的可验证答案。"
    assert decision.unknown_source_ids == unknown
    assert decision.uncited_paragraphs == uncited
    assert decision.citation_syntax_passed is False

    audit_projection = decision.to_dict()
    public_projection = decision.to_public_dict()
    assert audit_projection["raw_output"] == raw_output
    assert audit_projection["uncited_paragraphs"] == list(uncited)
    assert "raw_output" not in public_projection
    assert "uncited_paragraphs" not in public_projection
    assert public_projection["raw_output_included"] is False
    assert public_projection["audit_findings_included"] is False
    assert public_projection["response_text"] == decision.response_text
    if raw_output != decision.response_text:
        assert raw_output not in json.dumps(public_projection, ensure_ascii=False)


def test_public_projection_includes_only_publishable_response() -> None:
    raw_output = "必须先检查权限。[S1]"
    decision = evaluate_post_generation(_context(), raw_output)

    public_projection = decision.to_public_dict()

    assert decision.action is PublicationAction.PUBLISH
    assert public_projection["response_text"] == raw_output
    assert public_projection["public_decision_fingerprint"] == (
        decision.public_decision_fingerprint
    )
    assert decision.to_public_dict() == public_projection


def test_model_output_equal_to_rejection_text_still_fails_closed() -> None:
    policy = RAGPublicationPolicy()

    decision = evaluate_post_generation(_context(), policy.rejected_response)

    assert decision.action is PublicationAction.REJECT
    assert decision.raw_output == decision.response_text
    assert decision.reason_code == "missing_citation"


def test_inconsistent_context_is_rejected_before_generation() -> None:
    document = Document("doc", "evidence", "tenant-a")
    context = CitationContext(rendered="", sources={"S1": document})
    calls = 0

    def generator(_: str) -> str:
        nonlocal calls
        calls += 1
        return "answer [S1]"

    with pytest.raises(ValueError, match="must have rendered context"):
        guard_rag_generation(context, generator)
    assert calls == 0

    with pytest.raises(ValueError, match="empty source map"):
        evaluate_pre_generation(CitationContext(rendered="orphan", sources={}))


def test_generation_type_size_and_exception_paths_fail_closed() -> None:
    context = _context()
    non_string: Any = 7
    with pytest.raises(TypeError, match="raw_output must be a string"):
        evaluate_post_generation(context, non_string)
    with pytest.raises(ValueError, match="character limit"):
        evaluate_post_generation(context, "x" * 4097)
    with pytest.raises(ValueError, match="before generation"):
        evaluate_post_generation(
            build_citation_context([], tenant_id="tenant-a"), "answer"
        )

    calls = 0

    def failing_generator(_: str) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        guard_rag_generation(context, failing_generator)
    assert calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"revision": "unknown"},
        {"no_evidence_response": ""},
        {"rejected_response": " "},
        {"no_evidence_response": "same", "rejected_response": "same"},
    ],
)
def test_publication_policy_rejects_ambiguous_configuration(
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        RAGPublicationPolicy(**kwargs)


def test_real_qwen_report_replays_as_reject_then_pre_generation_abstain(
    tmp_path: Path,
) -> None:
    spec, source_report = _verified_qwen_source()

    replay = build_publication_policy_replay_report(
        spec=spec,
        source_report=source_report,
    )
    path = tmp_path / "replay.json"
    _write_json(path, replay)
    verified = verify_publication_policy_replay_report(
        path,
        spec=spec,
        source_report=source_report,
    )
    frozen = verify_publication_policy_replay_report(
        FROZEN_REPLAY_REPORT,
        spec=spec,
        source_report=source_report,
    )

    assert verified["report_version"] == RAG_PUBLICATION_REPLAY_REPORT_VERSION
    assert frozen == replay
    assert replay["report_fingerprint"] == (
        "sha256:ed4d16ad762d7cb8dbd66f8c51ce1ac4972c0f26679d7c36e085954a30b13239"
    )
    assert replay["source_rag_report_fingerprint"] == (
        "sha256:829663e216828ad418ddf9a6c38ee487fe44b38d3939072d0ce443e8e8ee5b60"
    )
    replay_policy = replay["policy"]
    assert isinstance(replay_policy, dict)
    assert replay_policy["revision"] == RAG_PUBLICATION_POLICY_VERSION
    assert replay["summary"] == {
        "case_count": 2,
        "publish_count": 0,
        "pre_generation_abstention_count": 1,
        "post_generation_rejection_count": 1,
        "unsafe_baseline_outputs_published_count": 0,
    }
    replay_cases = replay["cases"]
    assert isinstance(replay_cases, list)
    answerable, no_evidence = replay_cases
    assert isinstance(answerable, dict)
    assert isinstance(no_evidence, dict)
    answerable_decision = answerable["decision"]
    no_evidence_decision = no_evidence["decision"]
    assert isinstance(answerable_decision, dict)
    assert isinstance(no_evidence_decision, dict)
    assert answerable["policy_generator_call_count"] == 1
    assert answerable_decision["action"] == "reject"
    assert answerable_decision["reason_code"] == "missing_citation"
    raw_output = answerable_decision["raw_output"]
    assert isinstance(raw_output, str)
    assert raw_output.startswith("RAG 检索阶段")
    assert no_evidence["policy_generator_call_count"] == 0
    assert no_evidence_decision["action"] == "abstain"
    assert no_evidence_decision["raw_output"] is None
    replay_scope = replay["scope"]
    assert isinstance(replay_scope, dict)
    assert replay_scope["guarded_runtime_model_call_suppression_observed"] is False
    assert replay_scope["claim_evidence_entailment_verified"] is False
    assert "counterfactual policy replay" in RAG_PUBLICATION_REPLAY_EVIDENCE_BOUNDARY


def test_replay_rejects_source_report_identity_and_packing_drift() -> None:
    spec, source_report = _verified_qwen_source()

    stale_report = copy.deepcopy(source_report)
    stale_report["cases"][0]["generation"]["raw_output"] = "tampered"
    with pytest.raises(ValueError, match="source report fingerprint mismatch"):
        build_publication_policy_replay_report(
            spec=spec,
            source_report=stale_report,
        )

    stale_case = copy.deepcopy(source_report)
    stale_case["cases"][0]["generation"]["raw_output"] = "tampered"
    _rehash(stale_case, "report_fingerprint")
    with pytest.raises(ValueError, match=r"case_fingerprint mismatch"):
        build_publication_policy_replay_report(
            spec=spec,
            source_report=stale_case,
        )

    repacked = copy.deepcopy(source_report)
    repacked["cases"][0]["packing"]["document_ids"].reverse()
    _rehash(repacked["cases"][0], "case_fingerprint")
    _rehash(repacked, "report_fingerprint")
    with pytest.raises(ValueError, match="packing differs"):
        build_publication_policy_replay_report(
            spec=spec,
            source_report=repacked,
        )


def test_replay_verifier_rejects_extra_fields_duplicate_keys_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    spec, source_report = _verified_qwen_source()
    replay = build_publication_policy_replay_report(
        spec=spec,
        source_report=source_report,
    )

    extra = copy.deepcopy(replay)
    extra["unreviewed"] = True
    extra_path = tmp_path / "extra.json"
    _write_json(extra_path, extra)
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        verify_publication_policy_replay_report(
            extra_path,
            spec=spec,
            source_report=source_report,
        )

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"x": 1, "x": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        verify_publication_policy_replay_report(
            duplicate_path,
            spec=spec,
            source_report=source_report,
        )

    for index, raw in enumerate(('{"x": NaN}', '{"x": 1e999}')):
        path = tmp_path / f"nonfinite-{index}.json"
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError, match=r"non-finite|non-standard JSON"):
            verify_publication_policy_replay_report(
                path,
                spec=spec,
                source_report=source_report,
            )


