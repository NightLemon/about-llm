from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from about_llm.integrations.cloud_api_cli import (
    build_retry_matrix,
    load_contracts,
    main,
    verify_contracts,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "projects" / "cloud-api-contracts" / "contracts.example.jsonl"


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
