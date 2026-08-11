from __future__ import annotations

import json
from pathlib import Path

import pytest

from about_llm.synthetic_data_cli import main

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "synthetic-data-audit"


def test_cli_audits_fixture_and_writes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "report.json"
    exit_code = main(
        [
            "--records",
            str(PROJECT / "records.example.jsonl"),
            "--required-verifier",
            "schema",
            "--required-verifier",
            "grounding",
            "--known-parent-id",
            "real-anchor-001",
            "--mixture",
            str(PROJECT / "mixture.example.json"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert printed == saved
    assert saved["audit"]["candidate_count"] == 4
    assert saved["audit"]["eligible_count"] == 2
    assert saved["audit"]["eligible_unique_content_count"] == 1
    assert saved["audit"]["self_verified_record_ids"] == ["syn-002"]
    assert saved["audit"]["missing_verifier_record_ids"] == ["syn-003"]
    assert saved["audit"]["failed_verifier_record_ids"] == ["syn-004"]
    assert saved["audit"]["unresolved_parent_pairs"] == []
    assert saved["mixture"]["synthetic_fraction"] == pytest.approx(0.25)
    assert saved["mixture"]["exposures"][1]["expected_repetition_factor"] == 5
    assert "does not prove" in saved["evidence_boundary"]


def test_cli_rejects_unknown_parent_without_silently_resolving_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"record_id":"x","content":"x","parent_ids":["unknown"],'
        '"generator_revision":"g@1","prompt_revision":"p@1","generation_round":1,'
        '"verifications":[{"verifier_id":"v","revision":"v@1","passed":true}]}\n',
        encoding="utf-8",
    )

    assert main(["--records", str(path), "--required-verifier", "v"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["audit"]["unresolved_parent_pairs"] == [["x", "unknown"]]


def test_cli_rejects_malformed_verifier_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"record_id":"x","content":"x","parent_ids":["p"],'
        '"generator_revision":"g@1","prompt_revision":"p@1","generation_round":1,'
        '"verifications":[{"verifier_id":"v","revision":"v@1","passed":1}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        main(["--records", str(path), "--required-verifier", "v"])
    assert error.value.code == 2
    assert "passed must be a boolean" in capsys.readouterr().err
