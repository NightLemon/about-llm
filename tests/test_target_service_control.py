from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import uvicorn

import about_llm.inference.target_service_control as target_module
from about_llm.inference.openai_reference import (
    OPENAI_REFERENCE_SERVICE_VERSION,
    ChatCompletionRequest,
    GeneratedCompletion,
)
from about_llm.inference.target_service_control import (
    EXPECTED_PARAMETER_COUNT,
    TARGET_SERVICE_EVIDENCE_BOUNDARY,
    build_target_app,
    load_target_service_control_spec,
    run_live_target_service_control,
    verify_recorded_target_service_report,
)
from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)
from about_llm.llmops import artifact_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "inference-serving"
pytestmark = pytest.mark.contract
CONTROL = PROJECT / "qwen2.5-0.5b-service.control.json"
RECORDED_REPORT = PROJECT / "qwen2.5-0.5b-service.recorded-report.json"
CHECKPOINT_CONTROL = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)


def _reviewed() -> tuple[Any, Any]:
    checkpoint = load_checkpoint_control_spec(CHECKPOINT_CONTROL)
    spec = load_target_service_control_spec(CONTROL)
    return checkpoint, spec


def _rehash(report: dict[str, Any]) -> None:
    report.pop("report_fingerprint", None)
    report["report_fingerprint"] = "sha256:" + artifact_fingerprint(report)


class _FixtureTargetBackend:
    def __init__(self, spec: Any) -> None:
        self.model_id = spec.model_id
        self.backend_fingerprint = target_module._backend_fingerprint(spec)
        self.spec = spec
        self.calls = 0

    async def generate(self, request: ChatCompletionRequest) -> GeneratedCompletion:
        assert request.model == self.model_id
        self.calls += 1
        return GeneratedCompletion(
            text="2<|im_end|>",
            text_deltas=("2", "<|im_end|>"),
            prompt_token_count=self.spec.expected_prompt_token_count,
            completion_token_ids=self.spec.expected_completion_token_ids,
            finish_reason=self.spec.expected_finish_reason,
        )

    def audit_projection(self) -> Mapping[str, Any]:
        execution = None
        if self.calls:
            execution = {
                "prompt_token_count": self.spec.expected_prompt_token_count,
                "completion_token_ids": list(
                    self.spec.expected_completion_token_ids
                ),
                "completion_text_fingerprint": (
                    self.spec.expected_completion_text_fingerprint
                ),
                "finish_reason": self.spec.expected_finish_reason,
            }
        return {
            "implementation": "fixture.GenerationMixin.generate",
            "device": "cpu",
            "generation_call_count": self.calls,
            "last_execution": execution,
        }


class _ThreadedUvicornProcess:
    """Popen-shaped, real-TCP test double that never loads checkpoint weights."""

    def __init__(
        self,
        command: list[str],
        *,
        env: Mapping[str, str],
        spec: Any,
        checkpoint: Any,
        **_: Any,
    ) -> None:
        port = int(command[command.index("--port") + 1])
        token = env[target_module.CONTROL_TOKEN_ENV]
        files = [
            {
                "filename": item.filename,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
                "verified": True,
            }
            for item in checkpoint.files
        ]
        metadata = {
            "runtime": dict(spec.reviewed_runtime),
            "artifacts": {
                "selected_file_count": len(files),
                "selected_total_bytes": sum(
                    item.size_bytes for item in checkpoint.files
                ),
                "files": files,
            },
            "model_class": spec.expected_model_class,
            "model_type": spec.expected_model_type,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "parameters_frozen": True,
        }
        app = build_target_app(
            _FixtureTargetBackend(spec),  # type: ignore[arg-type]
            metadata,
            bearer_token=token,
            maximum_new_tokens=spec.max_tokens,
        )
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                access_log=False,
                log_level="critical",
                lifespan="off",
                ws="none",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()

    def poll(self) -> int | None:
        return None if self.thread.is_alive() else 0

    def terminate(self) -> None:
        self.server.should_exit = True

    def kill(self) -> None:
        self.server.force_exit = True
        self.server.should_exit = True

    def communicate(self, timeout: float) -> tuple[bytes, bytes]:
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise subprocess.TimeoutExpired("fixture-uvicorn", timeout)
        return b"", b""


def test_reviewed_manifest_binds_checkpoint_output_runtime_and_scope() -> None:
    checkpoint, spec = _reviewed()

    assert spec.manifest_fingerprint == (
        "sha256:cfb9b5409c1ccec7267d85e5adca2ae8f8e9e80c0ff4301f0414f659728fb4ea"
    )
    assert spec.checkpoint_manifest_fingerprint == checkpoint.manifest_fingerprint
    assert spec.expected_prompt_token_count == 31
    assert spec.expected_completion_token_ids == (17, 151645)
    assert spec.expected_completion_text_fingerprint == (
        "sha256:f734df76252d8e1047f3dcca7ecbcef3e8d07c1e24c28dd62eb023b88ffac4a5"
    )
    assert spec.expected_finish_reason == "stop"
    assert dict(spec.reviewed_runtime) == {
        "httpx": "0.28.1",
        "python": "3.12.10",
        "starlette": "0.41.3",
        "torch": "2.13.0+cpu",
        "transformers": "4.57.6",
        "uvicorn": "0.52.1",
    }
    assert "does not use vLLM, CUDA, TLS" in TARGET_SERVICE_EVIDENCE_BOUNDARY


def test_manifest_loader_rejects_unknown_duplicate_nonfinite_and_binding_drift(
    tmp_path: Path,
) -> None:
    manifest = json.loads(CONTROL.read_text(encoding="utf-8"))
    manifest["unknown"] = True
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fields are invalid"):
        load_target_service_control_spec(unknown)

    invalid_payloads = (
        b'{"control_version":"x","control_version":"y"}',
        b'{"value":NaN}',
        b"{\xff}",
    )
    for index, payload in enumerate(invalid_payloads):
        path = tmp_path / f"invalid-{index}.json"
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="strict UTF-8 JSON"):
            load_target_service_control_spec(path)

    checkpoint, spec = _reviewed()
    drifted = dataclasses.replace(checkpoint, revision="0" * 40)
    with pytest.raises(ValueError, match="checkpoint identity drift"):
        target_module._validate_checkpoint_binding(spec, drifted)


@pytest.mark.integration
@pytest.mark.extended
def test_fake_backend_exercises_real_tcp_control_and_offline_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, spec = _reviewed()

    def fake_popen(command: list[str], **kwargs: Any) -> _ThreadedUvicornProcess:
        return _ThreadedUvicornProcess(
            command,
            spec=spec,
            checkpoint=checkpoint,
            **kwargs,
        )

    monkeypatch.setattr(target_module.subprocess, "Popen", fake_popen)
    report = run_live_target_service_control(
        CONTROL,
        CHECKPOINT_CONTROL,
        local_files_only=True,
        request_timeout_seconds=10,
        server_start_timeout_seconds=10,
    )

    assert report["api"] == {
        "service_version": OPENAI_REFERENCE_SERVICE_VERSION,
        "models_endpoint_executed": True,
        "chat_nonstream_executed": True,
        "chat_sse_executed": True,
        "unauthorized_status": 401,
        "unknown_field_status": 422,
        "wrong_model_status": 404,
        "sse_done_observed": True,
        "stream_usage_observed": True,
        "raw_request_or_response_published": False,
    }
    assert report["execution"]["framework_generate_call_count"] == 2
    assert report["execution"]["stream_content_delta_count"] == 2
    assert report["server_process"]["stdout_stderr_empty"] is True
    assert report["scope"]["target_checkpoint_weights_loaded"] is True
    assert report["scope"]["vllm_executed"] is False
    assert verify_recorded_target_service_report(spec, checkpoint, report) == report


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["scope"].update({"cuda_executed": True}), "scope"),
        (
            lambda report: report["execution"].update({"parameter_count": 1}),
            "execution",
        ),
        (
            lambda report: report["api"].update({"wrong_model_status": 200}),
            "API",
        ),
        (
            lambda report: report["artifacts"]["files"][0].update(
                {"sha256": "sha256:" + "0" * 64}
            ),
            "artifact",
        ),
    ],
)
def test_offline_verifier_rejects_cooperatively_rehashed_drift(
    mutation: Any,
    message: str,
) -> None:
    checkpoint, spec = _reviewed()
    drifted = copy.deepcopy(json.loads(RECORDED_REPORT.read_text(encoding="utf-8")))
    mutation(drifted)
    _rehash(drifted)
    with pytest.raises(ValueError, match=message):
        verify_recorded_target_service_report(spec, checkpoint, drifted)


def test_live_control_rejects_nonpositive_timeouts_without_loading_weights() -> None:
    with pytest.raises(ValueError, match="timeouts must be positive"):
        run_live_target_service_control(
            CONTROL,
            CHECKPOINT_CONTROL,
            request_timeout_seconds=0,
        )


def test_recorded_real_qwen_report_verifies_without_loading_weights() -> None:
    checkpoint, spec = _reviewed()
    report = target_module.load_and_verify_recorded_target_service_report(
        CONTROL,
        CHECKPOINT_CONTROL,
        RECORDED_REPORT,
    )

    assert report["report_fingerprint"] == (
        "sha256:63e566ca60126c09c0f97f23b591e879d6efe7991b646f72bcc96ec493617ddb"
    )
    assert report["artifacts"]["selected_total_bytes"] == 999_586_347
    assert report["execution"]["completion_token_ids"] == [17, 151645]
    assert report["execution"]["framework_generate_call_count"] == 2
    assert report["scope"]["real_ipv4_loopback_tcp_http_executed"] is True
    assert report["scope"]["performance_capacity_or_slo_proven"] is False
    assert verify_recorded_target_service_report(spec, checkpoint, report) == report

