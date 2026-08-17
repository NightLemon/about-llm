from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from about_llm.inference.transformers_thread_cancellation_control import (
    BACKEND_FINGERPRINT,
    EXPECTED_PARAMETER_COUNT,
    REVIEWED_CHECKED_AT,
    REVIEWED_RUNTIME,
    TRANSFORMERS_THREAD_CANCELLATION_EVIDENCE_BOUNDARY,
    get_transformers_thread_cancellation_runtime,
    load_and_verify_transformers_thread_cancellation_report,
    run_transformers_thread_cancellation_control,
    serve_control,
    validate_transformers_thread_cancellation_observation,
    verify_transformers_thread_cancellation_report,
)
from about_llm.llmops import artifact_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "inference-serving"
RECORDED_REPORT = (
    PROJECT / "transformers-thread-cancellation.recorded-report.json"
)


def _rehash(report: dict[str, Any]) -> None:
    report.pop("report_fingerprint", None)
    report["report_fingerprint"] = "sha256:" + artifact_fingerprint(report)


def test_real_loopback_tiny_transformers_thread_cancellation_control() -> None:
    report = run_transformers_thread_cancellation_control()
    live_runtime = get_transformers_thread_cancellation_runtime()

    assert report["runtime"] == live_runtime
    assert report["checked_at"] == datetime.now(timezone.utc).date().isoformat()
    assert report["model"]["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert report["preclose"] == {
        "service_active_streams": 1,
        "service_cancelled_streams": 0,
        "backend_stream_call_count": 1,
        "generation_thread_alive": True,
        "generation_returned": False,
        "streamer_waiting_for_cancel": True,
        "stopping_criteria_observed_cancel": False,
        "generated_token_ids": [7],
        "forward_call_count": 1,
    }
    assert report["postclose"]["stopping_criteria_observed_cancel"] is True
    assert report["postclose"]["generation_thread_joined"] is True
    assert report["postclose"]["generation_thread_alive"] is False
    assert report["postclose"]["generated_token_ids"] == [7]
    assert report["postclose"]["forward_call_count"] == 1
    assert report["scope"]["transformers_generation_mixin_generate_executed"] is True
    assert report["scope"]["unmodified_transformers_cancellation_proven"] is False
    assert (
        validate_transformers_thread_cancellation_observation(
            report,
            expected_runtime=live_runtime,
            expected_checked_at=report["checked_at"],
        )
        == report
    )
    if (
        report["runtime"] != REVIEWED_RUNTIME
        or report["checked_at"] != REVIEWED_CHECKED_AT
    ):
        with pytest.raises(ValueError, match="identity/runtime/evidence boundary"):
            verify_transformers_thread_cancellation_report(report)


def test_recorded_report_verifies_without_constructing_model() -> None:
    report = load_and_verify_transformers_thread_cancellation_report(RECORDED_REPORT)

    assert report["report_fingerprint"] == (
        "sha256:eadcab544cc78dabfc171446fd825992cc1c12edbbc478679c8bb10f7cf62bc7"
    )
    assert report["backend_fingerprint"] == BACKEND_FINGERPRINT
    assert report["evidence_boundary"] == (
        TRANSFORMERS_THREAD_CANCELLATION_EVIDENCE_BOUNDARY
    )
    assert report["scope"]["kv_cpu_or_gpu_memory_release_proven"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["scope"].update(
            {"unmodified_transformers_cancellation_proven": True}
        ),
        lambda report: report["postclose"].update(
            {"generation_thread_joined": False}
        ),
        lambda report: report["postclose"].update({"generated_token_ids": [7, 7]}),
        lambda report: report["model"].update({"parameter_count": 1}),
        lambda report: report["runtime"].update({"python": "0.0.0"}),
        lambda report: report.update({"unknown": True}),
    ],
)
def test_verifier_rejects_cooperatively_rehashed_semantic_drift(mutate: Any) -> None:
    report = copy.deepcopy(json.loads(RECORDED_REPORT.read_text(encoding="utf-8")))
    mutate(report)
    _rehash(report)
    with pytest.raises(ValueError):
        verify_transformers_thread_cancellation_report(report)


def test_loader_rejects_duplicate_nonfinite_invalid_utf8_and_size(
    tmp_path: Path,
) -> None:
    invalid_payloads = (
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b"{\xff}",
        b"x" * 64_001,
    )
    for index, payload in enumerate(invalid_payloads):
        path = tmp_path / f"invalid-{index}.json"
        path.write_bytes(payload)
        with pytest.raises(ValueError):
            load_and_verify_transformers_thread_cancellation_report(path)


def test_serve_rejects_nonloopback_and_invalid_port_before_model_construction() -> None:
    with pytest.raises(ValueError, match="loopback"):
        serve_control("0.0.0.0", 1234, bearer_token="t" * 32)
    with pytest.raises(ValueError, match="port"):
        serve_control("127.0.0.1", 0, bearer_token="t" * 32)


def test_project_cli_verifies_report_without_constructing_model() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "transformers_thread_cancellation_control.py"),
            "--verify",
            str(RECORDED_REPORT),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["report_fingerprint"].endswith("10f7cf62bc7")
    assert report["scope"]["vllm_or_cuda_executed"] is False
