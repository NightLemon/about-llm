from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
    CheckpointFileEvidence,
    CheckpointMessage,
    VerifiedCheckpointSnapshot,
    load_checkpoint_control_spec,
)
from about_llm.llmops import artifact_fingerprint
from about_llm.rag.transformers_control import (
    RAG_TRANSFORMERS_EVIDENCE_BOUNDARY,
    execute_loaded_rag_transformers_control,
    load_rag_transformers_control_spec,
    validate_checkpoint_binding,
    verify_recorded_rag_transformers_report,
)

pytestmark = [pytest.mark.contract, pytest.mark.security, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "projects" / "rag-foundations" / "qwen2.5-0.5b-rag.control.json"
CHECKPOINT_MANIFEST = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)
SCRIPT = ROOT / "projects" / "rag-foundations" / "run_qwen_rag_control.py"
RECORDED_REPORT = (
    ROOT
    / "projects"
    / "rag-foundations"
    / "qwen2.5-0.5b-rag.recorded-report.json"
)


class TinyTokenizer:
    chat_template = "tiny reviewed chat template"
    pad_token_id = 0
    eos_token_id = 4

    def __len__(self) -> int:
        return 32

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> Any:
        import torch

        assert [message["role"] for message in messages] == ["system", "user"]
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert clean_up_tokenization_spaces is False
        values = [value for value in token_ids if not skip_special_tokens or value != 4]
        return " ".join("<eos>" if value == 4 else str(value) for value in values)


def _tiny_model() -> Any:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(23)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=4,
            pad_token_id=0,
        )
    )


def _tiny_bundle() -> tuple[Any, CheckpointControlSpec, VerifiedCheckpointSnapshot]:
    reviewed = load_rag_transformers_control_spec(MANIFEST)
    fingerprint = "sha256:" + "a" * 64
    spec = replace(
        reviewed,
        checkpoint_manifest_fingerprint=fingerprint,
        model_id="fixture/tiny-gpt2",
        revision="a" * 40,
        expected_model_class="GPT2LMHeadModel",
        expected_model_type="gpt2",
        max_new_tokens=4,
        prompt_budget_tokens=16,
        manifest_fingerprint="sha256:" + "b" * 64,
    )
    file = CheckpointFileEvidence(
        filename="model.safetensors",
        size_bytes=7,
        sha256="sha256:" + "c" * 64,
    )
    checkpoint = CheckpointControlSpec(
        checked_at="2026-08-13",
        model_id="fixture/tiny-gpt2",
        revision="a" * 40,
        source_base_url=(
            "https://huggingface.co/fixture/tiny-gpt2/resolve/" + "a" * 40 + "/"
        ),
        expected_model_class="GPT2LMHeadModel",
        expected_model_type="gpt2",
        dtype="float32",
        device="cpu",
        attention_implementation="eager",
        max_new_tokens=2,
        messages=(CheckpointMessage(role="user", content="fixture"),),
        files=(file,),
        manifest_fingerprint=fingerprint,
    )
    snapshot = VerifiedCheckpointSnapshot(
        directory=ROOT,
        files=(
            {
                "filename": file.filename,
                "size_bytes": file.size_bytes,
                "sha256": file.sha256,
                "verified": True,
            },
        ),
    )
    return spec, checkpoint, snapshot


def _write_report(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def _rehash(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    del unsigned["report_fingerprint"]
    value["report_fingerprint"] = "sha256:" + artifact_fingerprint(unsigned)


def _execute_tiny() -> tuple[dict[str, object], Any, CheckpointControlSpec]:
    spec, checkpoint, snapshot = _tiny_bundle()
    report = execute_loaded_rag_transformers_control(
        spec,
        checkpoint_spec=checkpoint,
        snapshot=snapshot,
        model=_tiny_model(),
        tokenizer=TinyTokenizer(),
    )
    return report, spec, checkpoint


def test_reviewed_manifest_binds_checkpoint_retrieval_and_failure_contract() -> None:
    spec = load_rag_transformers_control_spec(MANIFEST)
    checkpoint = load_checkpoint_control_spec(CHECKPOINT_MANIFEST)
    validate_checkpoint_binding(spec, checkpoint)

    assert spec.manifest_fingerprint == (
        "sha256:4ee166171982118552fcc73e38902e653596b652fc645573583a5ef6ca609dfd"
    )
    assert [case.expected_behavior for case in spec.cases] == [
        "answer_with_citations",
        "abstain",
    ]
    assert spec.cases[0].expected_retrieved_document_ids == (
        "acl-order-v1",
        "citation-boundary-v1",
    )
    assert spec.cases[1].expected_retrieved_document_ids == ()
    assert "records model failures" in RAG_TRANSFORMERS_EVIDENCE_BOUNDARY


def test_tiny_real_transformers_path_executes_retrieval_packing_and_greedy() -> None:
    report, _, _ = _execute_tiny()

    assert report["summary"] == {
        "case_count": 2,
        "expected_behavior_gate_passed_count": 0,
        "all_expected_behavior_gates_passed": False,
    }
    for case in report["cases"]:
        assert case["generation"]["manual_greedy_matches_generate"] is True
        assert len(case["generation"]["greedy_step_logits_sha256"]) == len(
            case["generation"]["generated_token_ids"]
        )
        assert case["verification"]["expected_behavior_gate_passed"] is False
    assert report["scope"]["claim_evidence_entailment_verified"] is False


def test_recorded_qwen_attempt_preserves_citation_and_abstention_failures() -> None:
    spec = load_rag_transformers_control_spec(MANIFEST)
    checkpoint = load_checkpoint_control_spec(CHECKPOINT_MANIFEST)
    report = verify_recorded_rag_transformers_report(
        RECORDED_REPORT, spec=spec, checkpoint_spec=checkpoint
    )

    assert report["report_fingerprint"] == (
        "sha256:829663e216828ad418ddf9a6c38ee487fe44b38d3939072d0ce443e8e8ee5b60"
    )
    assert report["summary"] == {
        "all_expected_behavior_gates_passed": False,
        "case_count": 2,
        "expected_behavior_gate_passed_count": 0,
    }
    answerable, no_answer = report["cases"]
    assert answerable["generation"]["generated_ended_with_eos"] is True
    assert answerable["verification"]["citation_syntax_passed"] is False
    assert no_answer["packing"]["document_ids"] == []
    assert no_answer["generation"]["stop_reason"] == "max_new_tokens"
    assert no_answer["verification"]["abstention_exact_match"] is False


def test_generated_report_round_trip_verifier_preserves_observed_failure(
    tmp_path: Path,
) -> None:
    report, spec, checkpoint = _execute_tiny()
    path = tmp_path / "report.json"
    _write_report(path, report)

    loaded = verify_recorded_rag_transformers_report(
        path, spec=spec, checkpoint_spec=checkpoint
    )

    assert loaded["report_fingerprint"] == report["report_fingerprint"]
    assert loaded["summary"]["all_expected_behavior_gates_passed"] is False


def test_report_nested_schema_rejects_cooperatively_rehashed_extra_field(
    tmp_path: Path,
) -> None:
    report, spec, checkpoint = _execute_tiny()
    report["cases"][0]["generation"]["unreviewed"] = True
    case = report["cases"][0]
    case_unsigned = dict(case)
    del case_unsigned["case_fingerprint"]
    case["case_fingerprint"] = "sha256:" + artifact_fingerprint(case_unsigned)
    _rehash(report)
    path = tmp_path / "report.json"
    _write_report(path, report)

    with pytest.raises(ValueError, match=r"generation: field set mismatch"):
        verify_recorded_rag_transformers_report(
            path, spec=spec, checkpoint_spec=checkpoint
        )


def test_report_cooperative_raw_output_rehash_cannot_hide_stale_local_audit(
    tmp_path: Path,
) -> None:
    report, spec, checkpoint = _execute_tiny()
    case = report["cases"][0]
    generation = case["generation"]
    generation["raw_output"] = "fabricated [S1]"
    generation["raw_output_sha256"] = (
        "sha256:" + __import__("hashlib").sha256(b"fabricated [S1]").hexdigest()
    )
    case_unsigned = dict(case)
    del case_unsigned["case_fingerprint"]
    case["case_fingerprint"] = "sha256:" + artifact_fingerprint(case_unsigned)
    _rehash(report)
    path = tmp_path / "report.json"
    _write_report(path, report)

    with pytest.raises(ValueError, match=r"verification\.cited_source_ids mismatch"):
        verify_recorded_rag_transformers_report(
            path, spec=spec, checkpoint_spec=checkpoint
        )


def test_report_cooperative_rehash_cannot_hide_bm25_score_drift(
    tmp_path: Path,
) -> None:
    report, spec, checkpoint = _execute_tiny()
    case = report["cases"][0]
    case["retrieval"]["results"][0]["score"] += 1.0
    case_unsigned = dict(case)
    del case_unsigned["case_fingerprint"]
    case["case_fingerprint"] = "sha256:" + artifact_fingerprint(case_unsigned)
    _rehash(report)
    path = tmp_path / "report.json"
    _write_report(path, report)

    with pytest.raises(ValueError, match="score differs from BM25 reconstruction"):
        verify_recorded_rag_transformers_report(
            path, spec=spec, checkpoint_spec=checkpoint
        )


def test_manifest_rejects_unknown_nested_field(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["generation"]["temperature"] = 0
    path = tmp_path / "manifest.json"
    _write_report(path, value)

    with pytest.raises(ValueError, match=r"manifest.generation: field set mismatch"):
        load_rag_transformers_control_spec(path)
