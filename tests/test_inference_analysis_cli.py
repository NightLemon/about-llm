from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from about_llm.inference_analysis_cli import main

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = ROOT / "projects" / "inference-serving" / "attempts.example.jsonl"
MANIFEST = ROOT / "projects" / "inference-serving" / "attempts.manifest.example.json"


@pytest.mark.smoke
def test_cli_reports_attempt_reliability_and_success_latency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "report.json"
    exit_code = main(
        [
            "--attempts",
            str(ATTEMPTS),
            "--benchmark-started-at",
            "0",
            "--benchmark-completed-at",
            "2",
            "--minimum-success-rate",
            "0.75",
            "--maximum-ttft-p95",
            "0.5",
            "--maximum-e2e-p95",
            "1.5",
            "--maximum-tpot-p95",
            "0.3",
            "--maximum-client-queue-p95",
            "0.2",
            "--maximum-successful-offered-ttft-p95",
            "0.6",
            "--maximum-offered-to-terminal-p95",
            "1.5",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert printed == saved
    assert saved["passed"] is True
    assert saved["summary"]["attempted_requests"] == 4
    assert saved["summary"]["successful_requests"] == 3
    assert saved["summary"]["success_rate"] == pytest.approx(0.75)
    assert saved["summary"]["failure_counts"] == {"rate_limited": 1}
    assert saved["summary"]["offered_timing_attempt_count"] == 4
    assert saved["summary"]["client_queue_p95_seconds"] == pytest.approx(0.185)
    assert saved["summary"]["successful_offered_ttft_p95_seconds"] == pytest.approx(
        0.58
    )
    assert saved["summary"]["offered_to_terminal_p95_seconds"] == pytest.approx(
        1.37
    )
    assert "recorded client attempts only" in saved["evidence_boundary"]


def test_cli_failure_exit_retains_every_gate_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--attempts",
            str(ATTEMPTS),
            "--benchmark-started-at",
            "0",
            "--benchmark-completed-at",
            "2",
            "--minimum-success-rate",
            "1",
            "--maximum-ttft-p95",
            "0.3",
            "--maximum-e2e-p95",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["passed"] is False
    assert len(payload["reasons"]) == 3


def test_cli_rejects_success_row_without_token_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"request_id":"x","outcome":"success","started_at":0,'
        '"first_token_at":0.1,"completed_at":1}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--attempts",
                str(path),
                "--benchmark-started-at",
                "0",
                "--benchmark-completed-at",
                "2",
                "--minimum-success-rate",
                "1",
            ]
        )
    assert error.value.code == 2
    assert "requires prompt_tokens" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            '{"request_id":"x","request_id":"y","outcome":"timeout",'
            '"started_at":0,"completed_at":1}',
            "duplicate JSON object key",
        ),
        (
            '{"request_id":"x","outcome":"timeout","started_at":0,'
            '"completed_at":1,"surprise":true}',
            "unknown=['surprise']",
        ),
        (
            '{"request_id":"x","outcome":"timeout","started_at":1e400,'
            '"completed_at":1}',
            "started_at must be finite",
        ),
    ],
)
def test_cli_rejects_ambiguous_attempt_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    payload: str,
    expected: str,
) -> None:
    path = tmp_path / "bad-artifact.jsonl"
    path.write_text(payload + "\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--attempts",
                str(path),
                "--benchmark-started-at",
                "0",
                "--benchmark-completed-at",
                "2",
                "--minimum-success-rate",
                "0",
            ]
        )

    assert error.value.code == 2
    assert expected in capsys.readouterr().err


def test_attempt_fixture_manifest_matches_recorded_artifact() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest = hashlib.sha256(ATTEMPTS.read_bytes()).hexdigest()

    assert manifest["fixture_only"] is True
    assert manifest["artifact_version"] == "fixture-v2"
    assert manifest["input"]["sha256"] == digest
    assert manifest["input"]["attempt_count"] == len(
        ATTEMPTS.read_text(encoding="utf-8").splitlines()
    )
    assert manifest["workload"]["hardware"] is None
    assert "not a GPU benchmark" in manifest["evidence_boundary"]
