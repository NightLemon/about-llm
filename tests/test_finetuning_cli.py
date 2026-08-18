from __future__ import annotations

import json
from pathlib import Path

import pytest

from about_llm.finetuning_cli import main

pytestmark = [pytest.mark.contract, pytest.mark.security]

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"


@pytest.mark.smoke
def test_cli_audits_fixture_and_writes_machine_readable_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "audit.json"
    exit_code = main(
        [
            "audit",
            "--jsonl",
            str(PROJECT / "audit.example.jsonl"),
            "--require-splits",
            "train,validation,test",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert printed == saved
    assert saved["gate_passed"] is True
    assert saved["record_count"] == 4
    assert saved["manifest_fingerprint"].startswith("sha256:")
    assert saved["scope"]["license_legality_verified"] is False


def test_cli_returns_gate_failure_without_hiding_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "audit",
            "--jsonl",
            str(PROJECT / "train.example.jsonl"),
        ]
    )

    assert exit_code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["gate_passed"] is False
    assert report["missing_splits"] == ["validation", "test"]


@pytest.mark.smoke
def test_cli_accepts_explicit_train_only_policy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "audit",
                "--jsonl",
                str(PROJECT / "train.example.jsonl"),
                "--require-splits",
                "train",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["required_splits"] == ["train"]


def test_cli_rejects_invalid_split_policy(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "audit",
                "--jsonl",
                str(PROJECT / "audit.example.jsonl"),
                "--require-splits",
                "train,dev",
            ]
        )
    assert error.value.code == 2
    assert "unknown split" in capsys.readouterr().err


def test_cli_rejects_malformed_jsonl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"one","id":"two"}\n', encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(["audit", "--jsonl", str(path)])
    assert error.value.code == 2
    assert "duplicate JSON object key" in capsys.readouterr().err


@pytest.mark.smoke
def test_near_audit_fixture_passes_with_explicit_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "near-audit.json"
    exit_code = main(
        [
            "near-audit",
            "--jsonl",
            str(PROJECT / "audit.example.jsonl"),
            "--profile",
            "nfc_whitespace",
            "--ngram-size",
            "5",
            "--threshold",
            "0.9",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate_passed"] is True
    near = payload["near_duplicate_audit"]
    assert near["record_pair_count"] == 5
    assert near["comparison_count"] == 15
    assert near["findings"] == []
    assert near["scope"]["semantic_equivalence_verified"] is False
    assert "not proof" in payload["evidence_boundary"]


def test_near_audit_returns_candidate_gate_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = (PROJECT / "audit.example.jsonl").read_text(encoding="utf-8").splitlines()
    train = json.loads(source[0])
    test = json.loads(source[3])
    test["messages"][0]["content"] = train["messages"][1]["content"] + "!"
    test["messages"][1]["content"] = train["messages"][2]["content"] + "。"
    path = tmp_path / "near.jsonl"
    path.write_text(
        "\n".join((source[0], source[1], source[2], json.dumps(test, ensure_ascii=False)))
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "near-audit",
            "--jsonl",
            str(path),
            "--profile",
            "nfc_whitespace",
            "--ngram-size",
            "3",
            "--threshold",
            "0.8",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["exact_audit"]["gate_passed"] is True
    findings = payload["near_duplicate_audit"]["findings"]
    assert {finding["view"] for finding in findings} >= {
        "user_content",
        "assistant_content",
    }
    assert all(
        finding["similarity"]
        == finding["intersection_size"] / finding["union_size"]
        for finding in findings
    )


def test_near_audit_rejects_invalid_numeric_policy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "near-audit",
                "--jsonl",
                str(PROJECT / "audit.example.jsonl"),
                "--profile",
                "nfc_whitespace",
                "--threshold",
                "nan",
            ]
        )
    assert error.value.code == 2
    assert "threshold must be finite" in capsys.readouterr().err


@pytest.mark.smoke
def test_prepare_training_writes_reloadable_minimal_readiness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "prepared"
    exit_code = main(
        [
            "prepare-training",
            "--train-jsonl",
            str(PROJECT / "train.example.jsonl"),
            "--audit-jsonl",
            str(PROJECT / "audit.example.jsonl"),
            "--profile",
            "nfc_whitespace",
            "--ngram-size",
            "5",
            "--threshold",
            "0.9",
            "--output-dir",
            str(output),
            "--governance-policy",
            str(PROJECT / "governance-policy.example.json"),
            "--governance-evaluated-at",
            "2026-08-06T12:00:00Z",
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    readiness = json.loads(
        (output / "sft-training-readiness.json").read_text(encoding="utf-8")
    )
    assert printed == readiness
    assert readiness["artifact_version"] == "about-llm.sft-training-readiness.v3"
    assert readiness["gate_passed"] is True
    assert readiness["near_duplicate_candidate_count"] == 0
    assert readiness["scope"]["held_out_plaintext_embedded"] is False
    assert (output / "sft-data-binding.json").exists()
    assert (output / "sft-near-duplicate-audit.json").exists()
    assert (output / "sft-governance-audit.json").exists()


@pytest.mark.smoke
def test_governance_audit_cli_is_explicitly_time_scoped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "governance.json"
    exit_code = main(
        [
            "governance-audit",
            "--jsonl",
            str(PROJECT / "audit.example.jsonl"),
            "--policy",
            str(PROJECT / "governance-policy.example.json"),
            "--evaluated-at",
            "2026-08-06T12:00:00Z",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate_passed"] is True
    assert payload["evaluated_at"] == "2026-08-06T12:00:00Z"
    assert payload["scope"]["legal_permission_verified"] is False
