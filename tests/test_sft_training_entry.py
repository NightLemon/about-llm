from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from about_llm.finetuning_cli import main as data_cli_main

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"


def _prepare(tmp_path: Path, combined: Path | None = None) -> Path:
    audit_output = tmp_path / "audit-output"
    exit_code = data_cli_main(
        [
            "prepare-training",
            "--train-jsonl",
            str(PROJECT / "train.example.jsonl"),
            "--audit-jsonl",
            str(combined or PROJECT / "audit.example.jsonl"),
            "--profile",
            "nfc_whitespace",
            "--ngram-size",
            "5",
            "--threshold",
            "0.9",
            "--output-dir",
            str(audit_output),
            "--governance-policy",
            str(PROJECT / "governance-policy.example.json"),
            "--governance-evaluated-at",
            "2026-08-06T12:00:00Z",
        ]
    )
    assert exit_code == 0
    return audit_output / "sft-training-readiness.json"


def _run_preflight(
    script: str,
    *,
    train_jsonl: Path,
    readiness_json: Path,
    output: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(PROJECT / script),
        "--model-id",
        "must-not-download",
        "--revision",
        "deadbeef",
        "--train-jsonl",
        str(train_jsonl),
        "--readiness-json",
        str(readiness_json),
        "--output-dir",
        str(output),
        "--data-preflight-only",
    ]
    if script == "train_qlora.py":
        command.extend(
            [
                "--num-parameters",
                "1000000",
                "--num-layers",
                "2",
                "--hidden-size",
                "64",
            ]
        )
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_training_entries_do_not_need_held_out_file_after_prepare(tmp_path: Path) -> None:
    combined = tmp_path / "combined.jsonl"
    combined.write_text(
        (PROJECT / "audit.example.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    readiness = _prepare(tmp_path, combined)
    combined.unlink()

    for script in ("train_trl_sft.py", "train_qlora.py"):
        output = tmp_path / script
        result = _run_preflight(
            script,
            train_jsonl=PROJECT / "train.example.jsonl",
            readiness_json=readiness,
            output=output,
        )
        assert result.returncode == 0, result.stderr
        assert (output / "sft-data-audit.json").exists()
        consumed = json.loads(
            (output / "sft-training-readiness.json").read_text(encoding="utf-8")
        )
        assert consumed["scope"]["trainer_needs_held_out_access"] is False
        assert not (output / "sft-template-mask-audit.json").exists()


def test_training_entry_rejects_tampered_readiness_before_model_load(
    tmp_path: Path,
) -> None:
    readiness = _prepare(tmp_path)
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    payload["near_duplicate_threshold"] = 0.8
    readiness.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "tampered-output"
    result = _run_preflight(
        "train_trl_sft.py",
        train_jsonl=PROJECT / "train.example.jsonl",
        readiness_json=readiness,
        output=output,
    )

    assert result.returncode != 0
    assert "manifest_fingerprint mismatch" in result.stderr
    assert not output.exists()


def test_training_entry_rejects_train_data_not_bound_by_readiness(tmp_path: Path) -> None:
    readiness = _prepare(tmp_path)
    changed = tmp_path / "changed-train.jsonl"
    lines = (PROJECT / "train.example.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["source"] = "different-source"
    changed.write_text(
        "\n".join((json.dumps(first, ensure_ascii=False), *lines[1:])) + "\n",
        encoding="utf-8",
    )

    result = _run_preflight(
        "train_qlora.py",
        train_jsonl=changed,
        readiness_json=readiness,
        output=tmp_path / "stale-output",
    )

    assert result.returncode != 0
    assert "ordered fingerprint differs" in result.stderr


def test_training_entry_rejects_valid_failed_gate_artifact(tmp_path: Path) -> None:
    source = (PROJECT / "audit.example.jsonl").read_text(encoding="utf-8").splitlines()
    train = json.loads(source[0])
    test = json.loads(source[3])
    test["messages"][0]["content"] = train["messages"][1]["content"] + "!"
    test["messages"][1]["content"] = train["messages"][2]["content"] + "."
    contaminated = tmp_path / "contaminated.jsonl"
    contaminated.write_text(
        "\n".join((source[0], source[1], source[2], json.dumps(test, ensure_ascii=False)))
        + "\n",
        encoding="utf-8",
    )
    audit_output = tmp_path / "failed-audit"
    exit_code = data_cli_main(
        [
            "prepare-training",
            "--train-jsonl",
            str(PROJECT / "train.example.jsonl"),
            "--audit-jsonl",
            str(contaminated),
            "--profile",
            "nfc_whitespace",
            "--ngram-size",
            "3",
            "--threshold",
            "0.8",
            "--output-dir",
            str(audit_output),
            "--governance-policy",
            str(PROJECT / "governance-policy.example.json"),
            "--governance-evaluated-at",
            "2026-08-06T12:00:00Z",
        ]
    )
    assert exit_code == 1

    result = _run_preflight(
        "train_trl_sft.py",
        train_jsonl=PROJECT / "train.example.jsonl",
        readiness_json=audit_output / "sft-training-readiness.json",
        output=tmp_path / "failed-output",
    )

    assert result.returncode != 0
    assert "readiness gate failed" in result.stderr


def test_training_entry_rejects_failed_governance_gate_artifact(tmp_path: Path) -> None:
    lines = (PROJECT / "audit.example.jsonl").read_text(encoding="utf-8").splitlines()
    validation = json.loads(lines[2])
    validation["license"] = "unregistered-license"
    combined = tmp_path / "unregistered.jsonl"
    combined.write_text(
        "\n".join(
            (
                lines[0],
                lines[1],
                json.dumps(validation, ensure_ascii=False),
                lines[3],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    audit_output = tmp_path / "governance-failed-audit"
    exit_code = data_cli_main(
        [
            "prepare-training",
            "--train-jsonl",
            str(PROJECT / "train.example.jsonl"),
            "--audit-jsonl",
            str(combined),
            "--profile",
            "nfc_whitespace",
            "--ngram-size",
            "5",
            "--threshold",
            "0.9",
            "--output-dir",
            str(audit_output),
            "--governance-policy",
            str(PROJECT / "governance-policy.example.json"),
            "--governance-evaluated-at",
            "2026-08-06T12:00:00Z",
        ]
    )
    assert exit_code == 1
    readiness_path = audit_output / "sft-training-readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["near_duplicate_candidate_count"] == 0
    assert readiness["governance_blocking_finding_count"] == 1

    result = _run_preflight(
        "train_trl_sft.py",
        train_jsonl=PROJECT / "train.example.jsonl",
        readiness_json=readiness_path,
        output=tmp_path / "governance-failed-output",
    )

    assert result.returncode != 0
    assert "1 governance finding(s)" in result.stderr
