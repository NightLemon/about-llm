from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from about_llm.integrations.cloud_api_cli import (
    build_reasoning_replay_matrix,
    build_retry_matrix,
    load_contracts,
    load_trajectory_release_candidates,
    main,
    verify_contracts,
)

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "projects" / "cloud-api-contracts" / "contracts.example.jsonl"
RELEASE_TRAJECTORY = (
    ROOT / "projects" / "cloud-api-contracts" / "trajectory-release.example.json"
)


def test_contract_fixtures_build_parse_and_redact_credentials() -> None:
    report = verify_contracts(load_contracts(CONTRACTS))
    rendered = json.dumps(report, ensure_ascii=False)

    assert report["passed"] is True
    assert report["network_performed"] is False
    assert report["real_credentials_used"] is False
    assert report["case_count"] == 3
    assert "offline-contract-secret-never-sent" not in rendered
    assert all(
        "<redacted>" in case["request"]["headers"].values()
        for case in report["cases"]
    )


def test_contract_mismatch_fails_without_hiding_actual_response() -> None:
    case = load_contracts(CONTRACTS)[0]
    report = verify_contracts([replace(case, expected={"text": "wrong"})])

    assert report["passed"] is False
    assert "expected 'wrong'" in report["cases"][0]["mismatches"][0]
    assert report["cases"][0]["parsed_response"]["text"] == "retrieval augmented generation"


def test_cli_writes_offline_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "contracts.json"

    exit_code = main(
        ["verify", "--contracts", str(CONTRACTS), "--output", str(output)]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["passed"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_retry_matrix_is_offline_and_fail_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    matrix = build_retry_matrix()
    assert matrix["passed"] is True
    assert matrix["network_performed"] is False
    reasons = {row["decision"]["reason"] for row in matrix["cases"]}
    assert {"replay_unsafe", "outcome_uncertain", "retry_after_too_long"} <= reasons
    malformed = next(
        row for row in matrix["cases"] if row["case_id"] == "malformed-retry-after-fallback"
    )
    assert malformed["decision"]["retry_after_state"] == "malformed"

    output = tmp_path / "retry-matrix.json"
    assert main(["retry-matrix", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(
        output.read_text(encoding="utf-8")
    )


def test_reasoning_replay_matrix_exposes_weak_binding_without_plaintext(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    matrix = build_reasoning_replay_matrix()
    rendered = json.dumps(matrix, ensure_ascii=False)

    assert matrix["passed"] is True
    assert matrix["case_count"] == 16
    assert matrix["unsafe_acceptance_count"] == 4
    assert matrix["network_performed"] is False
    assert matrix["real_provider_artifacts_used"] is False
    assert matrix["plaintext_reasoning_emitted"] is False
    assert "authored local reasoning" not in rendered
    bound_failures = {
        row["actual_reason"]
        for row in matrix["cases"]
        if row["binding_mode"] == "context-bound" and not row["actual_accepted"]
    }
    assert {
        "subject_mismatch",
        "tenant_mismatch",
        "session_mismatch",
        "branch_mismatch",
        "predecessor_mismatch",
        "model_not_allowed",
        "expired",
        "retired_key",
        "authentication_failed",
        "replay_detected",
    } <= bound_failures

    output = tmp_path / "reasoning-replay-matrix.json"
    assert main(["reasoning-replay-matrix", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(
        output.read_text(encoding="utf-8")
    )


def test_trajectory_release_gate_cli_accepts_safe_projection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    candidates = load_trajectory_release_candidates(RELEASE_TRAJECTORY)
    assert len(candidates) == 1

    output = tmp_path / "trajectory-release-report.json"
    assert main(
        [
            "trajectory-release-gate",
            "--input",
            str(RELEASE_TRAJECTORY),
            "--output",
            str(output),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["opaque_reasoning_block_count"] == 0
    assert payload["unknown_block_count"] == 0
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_trajectory_release_gate_cli_rejects_opaque_data_without_echoing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "never-render-this-provider-payload"
    candidate = json.loads(RELEASE_TRAJECTORY.read_text(encoding="utf-8"))
    candidate["turns"][1]["blocks"].append(
        {"type": "encrypted_reasoning", "data": secret}
    )
    path = tmp_path / "unsafe.jsonl"
    path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")

    exit_code = main(["trajectory-release-gate", "--input", str(path)])
    rendered = capsys.readouterr().out

    assert exit_code == 1
    assert json.loads(rendered)["opaque_reasoning_block_count"] == 1
    assert secret not in rendered


@pytest.mark.parametrize(
    "mutation",
    [
        lambda line: line.replace('"case_id":', '"case_id": "duplicate", "case_id":', 1),
        lambda line: line.replace('"max_tokens":32', '"max_tokens":NaN', 1),
        lambda line: line[:-1] + ', "unexpected": true}',
    ],
)
def test_contract_loader_rejects_ambiguous_json_and_unknown_fields(
    tmp_path: Path, mutation: Callable[[str], str]
) -> None:
    first = CONTRACTS.read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "invalid.jsonl"
    path.write_text(mutation(first) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_contracts(path)
