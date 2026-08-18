from __future__ import annotations

import json
from pathlib import Path

import pytest

from about_llm.llmops import artifact_fingerprint
from about_llm.synthetic_data_cli import (
    SYNTHETIC_AUDIT_ARTIFACT_VERSION,
    main,
    verify_synthetic_audit_artifact,
)

pytestmark = [pytest.mark.contract, pytest.mark.security]

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "synthetic-data-audit"
VALID_RECORD = (
    '{"record_id":"x","content":"x","parent_ids":["parent"],'
    '"generator_revision":"g@1","prompt_revision":"p@1","generation_round":1,'
    '"verifications":[{"verifier_id":"v","revision":"v@1","passed":true}]}\n'
)


def _fixture_args(output: Path | None = None) -> list[str]:
    arguments = [
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
    ]
    if output is not None:
        arguments.extend(["--output", str(output)])
    return arguments


def test_recorded_fixture_passes_full_local_recomputation() -> None:
    report = verify_synthetic_audit_artifact(
        PROJECT / "audit.example.json",
        records_path=PROJECT / "records.example.jsonl",
        required_verifiers=("schema", "grounding"),
        known_parent_ids=("real-anchor-001",),
        mixture_path=PROJECT / "mixture.example.json",
    )

    assert report["schema_version"] == SYNTHETIC_AUDIT_ARTIFACT_VERSION
    assert report["report_fingerprint"] == (
        "sha256:202d8db97b704c5542e8516c5bd0c945"
        "da1c1022100f6ecbfb828f2d2bb6f4cd"
    )


@pytest.mark.smoke
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
    assert saved["schema_version"] == SYNTHETIC_AUDIT_ARTIFACT_VERSION
    assert saved["report_fingerprint"].startswith("sha256:")
    assert saved["inputs"]["records"]["size_bytes"] > 0
    assert saved["inputs"]["records"]["sha256"].startswith("sha256:")
    assert saved["inputs"]["mixture"]["size_bytes"] > 0
    assert saved["policy"] == {
        "duplicate_content_affects_verifier_eligibility": False,
        "fingerprint_profile": "byte_exact",
        "known_parent_ids": ["real-anchor-001"],
        "required_verifiers": ["grounding", "schema"],
        "revision_overlap_affects_verifier_eligibility": False,
        "unresolved_lineage_affects_verifier_eligibility": False,
    }
    assert saved["audit"]["candidate_count"] == 4
    assert saved["audit"]["eligible_count"] == 2
    assert saved["audit"]["eligible_unique_content_count"] == 1
    assert saved["audit"]["self_verified_record_ids"] == ["syn-002"]
    assert saved["audit"]["missing_verifier_record_ids"] == ["syn-003"]
    assert saved["audit"]["failed_verifier_record_ids"] == ["syn-004"]
    assert saved["audit"]["unresolved_parent_pairs"] == []
    assert saved["audit"]["nonmonotonic_parent_pairs"] == []
    assert saved["audit"]["lineage_cycle_record_ids"] == []
    assert saved["mixture"]["synthetic_fraction"] == pytest.approx(0.25)
    assert saved["mixture"]["exposures"][1]["expected_repetition_factor"] == 5
    assert saved["scope"]["input_bytes_and_external_policy_bound"] is True
    assert saved["scope"]["training_or_observed_token_ledger_executed"] is False
    assert "does not prove" in saved["evidence_boundary"]

    verify_exit_code = main(
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
            "--verify-report",
            str(output),
        ]
    )
    assert verify_exit_code == 0
    verification = json.loads(capsys.readouterr().out)
    assert verification == {
        "report_fingerprint": saved["report_fingerprint"],
        "schema_version": SYNTHETIC_AUDIT_ARTIFACT_VERSION,
        "verification_scope": "full_local_recomputation",
        "verified": True,
    }


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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            VALID_RECORD.replace(
                '"record_id":"x"',
                '"record_id":"x","record_id":"changed"',
            ),
            "duplicate JSON object key",
        ),
        (
            VALID_RECORD.replace(
                '"generation_round":1',
                '"generation_round":1,"unknown":true',
            ),
            "unknown=['unknown']",
        ),
        (
            VALID_RECORD.replace(
                '"passed":true',
                '"passed":true,"score":1',
            ),
            "unknown=['score']",
        ),
        (
            VALID_RECORD.replace('"generation_round":1', '"generation_round":NaN'),
            "non-standard JSON constant",
        ),
    ],
)
def test_cli_rejects_ambiguous_record_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    payload: str,
    message: str,
) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(payload, encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(["--records", str(records), "--required-verifier", "v"])

    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mixture_payload", "message"),
    [
        (
            '{"total_consumed_tokens":100,"total_consumed_tokens":200,'
            '"components":[]}',
            "duplicate JSON object key",
        ),
        (
            '{"total_consumed_tokens":100,"components":[{"name":"real",'
            '"source_kind":"real","unique_tokens":10,"weight":1,"extra":0}]}',
            "unknown=['extra']",
        ),
        (
            '{"total_consumed_tokens":100,"components":[{"name":"real",'
            '"source_kind":"real","unique_tokens":10,"weight":Infinity}]}',
            "non-standard JSON constant",
        ),
    ],
)
def test_cli_rejects_ambiguous_mixture_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mixture_payload: str,
    message: str,
) -> None:
    records = tmp_path / "records.jsonl"
    mixture = tmp_path / "mixture.json"
    records.write_text(VALID_RECORD, encoding="utf-8")
    mixture.write_text(mixture_payload, encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--records",
                str(records),
                "--required-verifier",
                "v",
                "--known-parent-id",
                "parent",
                "--mixture",
                str(mixture),
            ]
        )

    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_report_verifier_rejects_cooperative_rehash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "report.json"
    assert main(_fixture_args(output)) == 0
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["audit"]["eligible_count"] = 3
    unsigned = dict(payload)
    del unsigned["report_fingerprint"]
    payload["report_fingerprint"] = "sha256:" + artifact_fingerprint(unsigned)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        main([*_fixture_args(), "--verify-report", str(output)])

    assert error.value.code == 2
    assert "full local recomputation" in capsys.readouterr().err


def test_report_verifier_binds_exact_input_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = tmp_path / "records.jsonl"
    output = tmp_path / "report.json"
    records.write_bytes((PROJECT / "records.example.jsonl").read_bytes())
    arguments = [
        "--records",
        str(records),
        "--required-verifier",
        "schema",
        "--required-verifier",
        "grounding",
        "--known-parent-id",
        "real-anchor-001",
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    capsys.readouterr()
    records.write_bytes(records.read_bytes() + b"\n")

    with pytest.raises(SystemExit) as error:
        main(
            [
                *arguments[:-2],
                "--verify-report",
                str(output),
            ]
        )

    assert error.value.code == 2
    assert "full local recomputation" in capsys.readouterr().err


def test_output_is_exclusive_create_unless_overwrite_is_explicit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "report.json"
    assert main(_fixture_args(output)) == 0
    capsys.readouterr()

    with pytest.raises(SystemExit) as error:
        main(_fixture_args(output))
    assert error.value.code == 2
    assert "exist" in capsys.readouterr().err.lower()

    assert main([*_fixture_args(output), "--overwrite"]) == 0
