from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from about_llm.preference_cli import main as preference_cli_main

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"


def _prepare(tmp_path: Path) -> Path:
    output = tmp_path / "prepared"
    assert (
        preference_cli_main(
            [
                "prepare-training",
                "--train-jsonl",
                str(PROJECT / "preference.train.example.jsonl"),
                "--audit-jsonl",
                str(PROJECT / "preference.example.jsonl"),
                "--output-dir",
                str(output),
                "--profile",
                "nfc_whitespace",
                "--ngram-size",
                "5",
                "--threshold",
                "0.9",
                "--governance-policy",
                str(PROJECT / "governance-policy.example.json"),
                "--governance-evaluated-at",
                "2026-08-06T12:00:00Z",
            ]
        )
        == 0
    )
    return output / "preference-training-readiness.json"


def _run(tmp_path: Path, readiness: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT / "train_trl_dpo.py"),
            "--model-id",
            "must-not-download",
            "--revision",
            "deadbeef",
            "--train-jsonl",
            str(PROJECT / "preference.train.example.jsonl"),
            "--readiness-json",
            str(readiness),
            "--output-dir",
            str(tmp_path / "run"),
            "--data-preflight-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_dpo_entry_needs_no_held_out_file_after_prepare(tmp_path: Path) -> None:
    readiness = _prepare(tmp_path)

    result = _run(tmp_path, readiness)

    assert result.returncode == 0, result.stderr
    output = tmp_path / "run"
    assert (output / "preference-train-audit.json").exists()
    consumed = json.loads(
        (output / "preference-training-readiness.json").read_text(encoding="utf-8")
    )
    assert consumed["scope"]["trainer_needs_held_out_access"] is False
    assert not (output / "preference-tokenization-audit.json").exists()


def test_dpo_entry_rejects_tampered_readiness_before_download(tmp_path: Path) -> None:
    readiness = _prepare(tmp_path)
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    payload["binary_train_record_count"] = 99
    readiness.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(tmp_path, readiness)

    assert result.returncode != 0
    assert "manifest_fingerprint mismatch" in result.stderr
    assert not (tmp_path / "run").exists()


def test_dpo_entry_rejects_valid_failed_governance_readiness(tmp_path: Path) -> None:
    lines = (PROJECT / "preference.example.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    validation = json.loads(lines[2])
    validation["license"] = "unregistered-license"
    combined = tmp_path / "governance-failed.jsonl"
    combined.write_text(
        "\n".join((lines[0], lines[1], json.dumps(validation), lines[3])) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "failed-prepared"
    exit_code = preference_cli_main(
        [
            "prepare-training",
            "--train-jsonl",
            str(PROJECT / "preference.train.example.jsonl"),
            "--audit-jsonl",
            str(combined),
            "--output-dir",
            str(output),
            "--profile",
            "nfc_whitespace",
            "--ngram-size",
            "5",
            "--threshold",
            "0.9",
            "--governance-policy",
            str(PROJECT / "governance-policy.example.json"),
            "--governance-evaluated-at",
            "2026-08-06T12:00:00Z",
        ]
    )
    assert exit_code == 1
    readiness = output / "preference-training-readiness.json"
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    assert payload["governance_blocking_finding_count"] == 1

    result = _run(tmp_path, readiness)

    assert result.returncode != 0
    assert "1 governance finding(s)" in result.stderr
