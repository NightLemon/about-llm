from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import about_llm.rag.guarded_transformers_control as guarded_control
from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
    CheckpointFileEvidence,
    CheckpointMessage,
    VerifiedCheckpointSnapshot,
    load_checkpoint_control_spec,
)
from about_llm.llmops import artifact_fingerprint
from about_llm.rag.guarded_transformers_control import (
    RAG_GUARDED_TRANSFORMERS_EVIDENCE_BOUNDARY,
    execute_loaded_guarded_rag_transformers_control,
    load_guarded_rag_transformers_control_spec,
    verify_recorded_guarded_rag_transformers_report,
)
from about_llm.rag.transformers_control import load_rag_transformers_control_spec

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "rag-foundations"
MANIFEST = PROJECT / "qwen2.5-0.5b-rag.guarded.control.json"
FAILURE_MANIFEST = PROJECT / "qwen2.5-0.5b-rag.control.json"
CHECKPOINT_MANIFEST = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)
SCRIPT = PROJECT / "run_qwen_guarded_rag_control.py"
RECORDED_REPORT = PROJECT / "qwen2.5-0.5b-rag.guarded.recorded-report.json"


class TinyTokenizer:
    chat_template = "tiny guarded chat template"
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
        visible = [
            value for value in token_ids if not skip_special_tokens or value != 4
        ]
        return " ".join("<eos>" if value == 4 else str(value) for value in visible)


def _tiny_model() -> Any:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(29)
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
    reviewed = load_guarded_rag_transformers_control_spec(MANIFEST)
    checkpoint_fingerprint = "sha256:" + "a" * 64
    spec = replace(
        reviewed,
        checkpoint_manifest_fingerprint=checkpoint_fingerprint,
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
        manifest_fingerprint=checkpoint_fingerprint,
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


def _execute_tiny() -> tuple[dict[str, object], Any, CheckpointControlSpec, int]:
    spec, checkpoint, snapshot = _tiny_bundle()
    model = _tiny_model()
    with patch.object(model, "generate", wraps=model.generate) as generate_spy:
        report = execute_loaded_guarded_rag_transformers_control(
            spec,
            checkpoint_spec=checkpoint,
            snapshot=snapshot,
            model=model,
            tokenizer=TinyTokenizer(),
        )
    return report, spec, checkpoint, generate_spy.call_count


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def _rehash_case(case: dict[str, Any]) -> None:
    unsigned = dict(case)
    unsigned.pop("case_fingerprint", None)
    case["case_fingerprint"] = "sha256:" + artifact_fingerprint(unsigned)


def _rehash_report(report: dict[str, Any]) -> None:
    unsigned = dict(report)
    unsigned.pop("report_fingerprint", None)
    report["report_fingerprint"] = "sha256:" + artifact_fingerprint(unsigned)


def test_manifest_is_distinct_and_binds_the_reviewed_checkpoint() -> None:
    guarded = load_guarded_rag_transformers_control_spec(MANIFEST)
    failure = load_rag_transformers_control_spec(FAILURE_MANIFEST)
    checkpoint = load_checkpoint_control_spec(CHECKPOINT_MANIFEST)

    assert guarded.checkpoint_manifest_fingerprint == checkpoint.manifest_fingerprint
    assert {case.case_id for case in guarded.cases}.isdisjoint(
        case.case_id for case in failure.cases
    )
    assert {case.query for case in guarded.cases}.isdisjoint(
        case.query for case in failure.cases
    )
    assert guarded.cases[0].expected_retrieved_document_ids == (
        "acl-order-v1",
        "citation-boundary-v1",
    )
    assert guarded.cases[1].expected_retrieved_document_ids == ()
    assert "does not replay model generation" in RAG_GUARDED_TRANSFORMERS_EVIDENCE_BOUNDARY


def test_real_tiny_transformers_generate_is_called_once_and_short_circuited_once() -> None:
    report, _, _, framework_spy_calls = _execute_tiny()
    value: Any = report
    answerable, empty = value["cases"]

    assert framework_spy_calls == 1
    assert answerable["generation"]["generator_callback_invocation_count"] == 1
    assert answerable["generation"]["framework_generate_invocation_count"] == 1
    assert answerable["prompt"]["prompt_transmitted_to_model"] is True
    assert empty["generation"]["generator_callback_invocation_count"] == 0
    assert empty["generation"]["framework_generate_invocation_count"] == 0
    assert empty["generation"]["generated_token_ids"] == []
    assert empty["prompt"]["prompt_transmitted_to_model"] is False
    assert value["summary"] == {
        "case_count": 2,
        "framework_generate_invocation_count": 1,
        "publish_count": 0,
        "pre_generation_abstention_count": 1,
        "post_generation_rejection_count": 1,
        "public_raw_output_field_count": 0,
    }


def test_run_path_uses_verified_local_snapshot_and_reviewed_loader_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec, checkpoint, snapshot = _tiny_bundle()
    observed: dict[str, Any] = {}

    def fake_download(value: Any, *, local_files_only: bool) -> Path:
        assert value is checkpoint
        assert local_files_only is True
        return ROOT

    def fake_verify(value: Any, directory: Path) -> VerifiedCheckpointSnapshot:
        assert value is checkpoint
        assert directory == ROOT
        return snapshot

    def fake_tokenizer(directory: Path, **kwargs: Any) -> TinyTokenizer:
        observed["tokenizer"] = (directory, kwargs)
        return TinyTokenizer()

    def fake_model(directory: Path, **kwargs: Any) -> Any:
        observed["model"] = (directory, kwargs)
        return _tiny_model()

    monkeypatch.setattr(guarded_control, "download_checkpoint_snapshot", fake_download)
    monkeypatch.setattr(guarded_control, "verify_checkpoint_snapshot", fake_verify)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_model)

    report = guarded_control.run_guarded_rag_transformers_control(
        spec, checkpoint_spec=checkpoint, local_files_only=True
    )

    assert report["summary"]["framework_generate_invocation_count"] == 1
    assert observed["tokenizer"] == (
        ROOT,
        {
            "trust_remote_code": False,
            "local_files_only": True,
            "use_fast": True,
        },
    )
    model_directory, model_kwargs = observed["model"]
    assert model_directory == ROOT
    assert model_kwargs["trust_remote_code"] is False
    assert model_kwargs["local_files_only"] is True
    assert model_kwargs["use_safetensors"] is True
    assert model_kwargs["low_cpu_mem_usage"] is True
    assert model_kwargs["attn_implementation"] == "eager"


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("case_count", "exactly two cases"),
        ("behavior", "one case for each behavior"),
        ("model_class", "model class does not match"),
        ("model_type", "model_type does not match"),
        ("chat_template", "must provide a chat template"),
        ("negative_token", "integer pad_token_id"),
        ("outside_token", "outside tokenizer vocabulary"),
        ("model_dtype", "parameters must be CPU FP32"),
    ],
)
def test_execute_preflight_rejects_runtime_and_workload_drift(
    fault: str, message: str
) -> None:
    spec, checkpoint, snapshot = _tiny_bundle()
    model: Any = _tiny_model()
    tokenizer = TinyTokenizer()
    if fault == "case_count":
        spec = replace(spec, cases=spec.cases[:1])
    elif fault == "behavior":
        second = replace(spec.cases[1], expected_behavior="answer_with_citations")
        spec = replace(spec, cases=(spec.cases[0], second))
    elif fault == "model_class":
        model = object()
    elif fault == "model_type":
        model.config.model_type = "other"
    elif fault == "chat_template":
        tokenizer.chat_template = ""
    elif fault == "negative_token":
        tokenizer.pad_token_id = -1
    elif fault == "outside_token":
        tokenizer.pad_token_id = 32
    else:
        model.double()

    with pytest.raises(ValueError, match=message):
        execute_loaded_guarded_rag_transformers_control(
            spec,
            checkpoint_spec=checkpoint,
            snapshot=snapshot,
            model=model,
            tokenizer=tokenizer,
        )


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("retrieval", "reviewed guarded BM25 identity drift"),
        ("packing", "reviewed guarded packing identity drift"),
    ],
)
def test_execute_rejects_reviewed_retrieval_or_packing_identity_drift(
    drift: str, message: str
) -> None:
    spec, checkpoint, snapshot = _tiny_bundle()
    first = spec.cases[0]
    if drift == "retrieval":
        first = replace(
            first,
            expected_retrieved_document_ids=("citation-boundary-v1",),
            expected_packed_document_ids=("citation-boundary-v1",),
        )
    else:
        first = replace(
            first,
            expected_packed_document_ids=("acl-order-v1",),
        )
    spec = replace(spec, cases=(first, spec.cases[1]))

    with pytest.raises(RuntimeError, match=message):
        execute_loaded_guarded_rag_transformers_control(
            spec,
            checkpoint_spec=checkpoint,
            snapshot=snapshot,
            model=_tiny_model(),
            tokenizer=TinyTokenizer(),
        )


def test_public_projection_excludes_raw_output_and_audit_findings() -> None:
    report, _, _, _ = _execute_tiny()
    value: Any = report

    assert value["cases"][0]["decision"]["raw_output"] is not None
    for case in value["cases"]:
        public = case["public_decision"]
        assert "raw_output" not in public
        assert "uncited_paragraphs" not in public
        assert "unknown_source_ids" not in public
        assert public["raw_output_included"] is False
        assert public["audit_findings_included"] is False


def test_recorded_qwen_report_proves_the_guarded_invocation_boundary() -> None:
    spec = load_guarded_rag_transformers_control_spec(MANIFEST)
    checkpoint = load_checkpoint_control_spec(CHECKPOINT_MANIFEST)
    report = verify_recorded_guarded_rag_transformers_report(
        RECORDED_REPORT, spec=spec, checkpoint_spec=checkpoint
    )
    answerable, empty = report["cases"]

    assert report["report_fingerprint"] == (
        "sha256:00706d003921282625e7c8ad89291c64493d35c13faf4ad7e7553a1388f29ede"
    )
    assert answerable["generation"]["framework_generate_invocation_count"] == 1
    assert answerable["decision"]["reason_code"] == "missing_citation"
    assert answerable["public_decision"]["action"] == "reject"
    assert empty["generation"]["framework_generate_invocation_count"] == 0
    assert empty["decision"]["stage"] == "pre_generation"
    assert empty["public_decision"]["action"] == "abstain"


@pytest.mark.parametrize(
    ("field", "replacement", "message", "rehash"),
    [
        ("report_version", "unsupported", "version is unsupported", True),
        ("manifest_fingerprint", "sha256:" + "0" * 64, "manifest fingerprint mismatch", True),
        (
            "checkpoint_manifest_fingerprint",
            "sha256:" + "0" * 64,
            "checkpoint fingerprint mismatch",
            True,
        ),
        ("checked_at", "2026-08-12", "checked_at mismatch", True),
        ("evidence_boundary", "drift", "evidence boundary drift", True),
        ("report_fingerprint", "sha256:" + "0" * 64, "report fingerprint mismatch", False),
    ],
)
def test_recorded_report_rejects_top_level_identity_drift(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
    rehash: bool,
) -> None:
    spec = load_guarded_rag_transformers_control_spec(MANIFEST)
    checkpoint = load_checkpoint_control_spec(CHECKPOINT_MANIFEST)
    value = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))
    value[field] = replacement
    if rehash:
        _rehash_report(value)
    path = tmp_path / "report.json"
    _write_json(path, value)

    with pytest.raises(ValueError, match=message):
        verify_recorded_guarded_rag_transformers_report(
            path, spec=spec, checkpoint_spec=checkpoint
        )


def test_generated_report_round_trips_through_offline_verifier(tmp_path: Path) -> None:
    report, spec, checkpoint, _ = _execute_tiny()
    path = tmp_path / "report.json"
    _write_json(path, report)

    loaded = verify_recorded_guarded_rag_transformers_report(
        path, spec=spec, checkpoint_spec=checkpoint
    )

    assert loaded["report_fingerprint"] == report["report_fingerprint"]
    assert loaded["summary"]["framework_generate_invocation_count"] == 1


def test_nested_extra_field_is_rejected_after_cooperative_rehash(
    tmp_path: Path,
) -> None:
    report, spec, checkpoint, _ = _execute_tiny()
    value: Any = report
    case = value["cases"][0]
    case["generation"]["unreviewed"] = True
    _rehash_case(case)
    _rehash_report(value)
    path = tmp_path / "report.json"
    _write_json(path, value)

    with pytest.raises(ValueError, match=r"generation: field set mismatch"):
        verify_recorded_guarded_rag_transformers_report(
            path, spec=spec, checkpoint_spec=checkpoint
        )


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("raw_output", "decision differs from local reconstruction"),
        ("decision", "decision differs from local reconstruction"),
        ("count", "generation invocation count mismatch"),
        ("packing", "packing decision mismatch"),
        ("chat_template", "prompt binding mismatch"),
    ],
)
def test_cooperative_rehash_cannot_hide_semantic_drift(
    tmp_path: Path, drift: str, message: str
) -> None:
    report, spec, checkpoint, _ = _execute_tiny()
    value: Any = report
    case = value["cases"][0]
    if drift == "raw_output":
        case["generation"]["raw_output"] = "fabricated [S1]"
        case["generation"]["raw_output_sha256"] = (
            "sha256:" + sha256(b"fabricated [S1]").hexdigest()
        )
    elif drift == "decision":
        case["decision"]["action"] = "publish"
    elif drift == "count":
        case["generation"]["framework_generate_invocation_count"] = 0
    elif drift == "packing":
        case["packing"]["decisions"][0]["cost_if_selected_units"] = 17
    else:
        case["prompt"]["chat_template_sha256"] = "sha256:" + "d" * 64
    _rehash_case(case)
    _rehash_report(value)
    path = tmp_path / "report.json"
    _write_json(path, value)

    with pytest.raises(ValueError, match=message):
        verify_recorded_guarded_rag_transformers_report(
            path, spec=spec, checkpoint_spec=checkpoint
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            b'{"report_version":"first","report_version":"second"}',
            "duplicate JSON object key",
        ),
        (b'{"report_version":NaN}', "non-standard JSON constant"),
        (b"\xff", "not valid UTF-8"),
    ],
)
def test_report_loader_rejects_non_strict_json(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    spec, checkpoint, _ = _tiny_bundle()
    path = tmp_path / "report.json"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        verify_recorded_guarded_rag_transformers_report(
            path, spec=spec, checkpoint_spec=checkpoint
        )


def test_manifest_requires_policy_abstention_text(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["prompts"]["abstention_text"] = "different"
    path = tmp_path / "manifest.json"
    _write_json(path, value)

    with pytest.raises(ValueError, match="abstention text differs from the policy"):
        load_guarded_rag_transformers_control_spec(path)


def test_cli_help_does_not_load_or_download_model() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert "--local-files-only" in completed.stdout
    assert "GenerationMixin.generate" in completed.stdout
    assert completed.stderr == ""
