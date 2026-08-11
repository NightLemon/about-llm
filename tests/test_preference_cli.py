from __future__ import annotations

import json
from pathlib import Path

import pytest

from about_llm.preference_cli import main

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "projects" / "single-gpu-finetuning" / "preference.example.jsonl"
TRAIN_FIXTURE = (
    ROOT / "projects" / "single-gpu-finetuning" / "preference.train.example.jsonl"
)
JUDGMENT_FIXTURE = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "preference-judgments.example.jsonl"
)


def test_preference_cli_writes_machine_readable_audit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "audit.json"

    exit_code = main(["audit", "--jsonl", str(FIXTURE), "--output", str(output)])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(output.read_text(encoding="utf-8"))
    assert printed["gate_passed"] is True
    assert printed["label_counts"] == {"a": 1, "b": 2, "tie": 1}
    assert printed["scope"]["tie_and_invalid_labels_preserved"] is True


def test_preference_cli_returns_gate_failure_with_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    payload["candidate_b"] = payload["candidate_a"]
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    exit_code = main(
        ["audit", "--jsonl", str(path), "--require-splits", "train"]
    )

    assert exit_code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["identical_candidate_record_ids"] == ["pref-train-alpha"]


def test_preference_cli_rejects_invalid_split_policy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["audit", "--jsonl", str(FIXTURE), "--require-splits", "train,dev"])

    assert error.value.code == 2
    assert "unknown split" in capsys.readouterr().err


def test_preference_cli_prepares_held_out_free_training_envelope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "prepared"

    exit_code = main(
        [
            "prepare-training",
            "--train-jsonl",
            str(TRAIN_FIXTURE),
            "--audit-jsonl",
            str(FIXTURE),
            "--output-dir",
            str(output),
            "--profile",
            "nfc_whitespace",
            "--ngram-size",
            "5",
            "--threshold",
            "0.9",
            "--governance-policy",
            str(ROOT / "projects" / "single-gpu-finetuning" / "governance-policy.example.json"),
            "--governance-evaluated-at",
            "2026-08-06T12:00:00Z",
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["gate_passed"] is True
    assert printed["binary_train_record_count"] == 2
    assert printed["scope"]["trainer_needs_held_out_access"] is False
    assert printed["near_duplicate_candidate_count"] == 0
    assert printed["governance_blocking_finding_count"] == 0
    assert set(path.name for path in output.iterdir()) == {
        "preference-train-audit.json",
        "preference-split-audit.json",
        "preference-data-binding.json",
        "preference-near-duplicate-audit.json",
        "preference-governance-audit.json",
        "preference-training-readiness.json",
    }


def test_preference_prepare_emits_failed_lexical_readiness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    test_record = json.loads(lines[3])
    test_record["candidate_a"] = "good alpha answer!"
    contaminated = tmp_path / "contaminated.jsonl"
    contaminated.write_text(
        "\n".join((*lines[:3], json.dumps(test_record))) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "prepare-training",
            "--train-jsonl",
            str(TRAIN_FIXTURE),
            "--audit-jsonl",
            str(contaminated),
            "--output-dir",
            str(tmp_path / "failed"),
            "--profile",
            "nfc_whitespace",
            "--ngram-size",
            "3",
            "--threshold",
            "0.8",
            "--governance-policy",
            str(
                ROOT
                / "projects"
                / "single-gpu-finetuning"
                / "governance-policy.example.json"
            ),
            "--governance-evaluated-at",
            "2026-08-06T12:00:00Z",
        ]
    )

    assert exit_code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["near_duplicate_candidate_count"] > 0
    assert report["gate_passed"] is False


def test_preference_cli_evaluates_bound_raw_judgments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "judgment-report.json"

    exit_code = main(
        [
            "evaluate-judgments",
            "--cases-jsonl",
            str(FIXTURE),
            "--judgments-jsonl",
            str(JUDGMENT_FIXTURE),
            "--judgments-per-pair",
            "4",
            "--minimum-per-order",
            "2",
            "--bootstrap-samples",
            "2000",
            "--bootstrap-seed",
            "17",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert report["evaluation"]["pairwise_agreement_numerator"] == 7
    assert report["evaluation"]["pairwise_agreement_denominator"] == 12
    assert report["evaluation"]["mean_pair_position_effect"] == 0.5
    assert report["evaluation"]["scope"]["causal_position_bias_identified"] is False


def test_preference_cli_withholds_statistics_when_judgment_gate_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text(
        "\n".join(JUDGMENT_FIXTURE.read_text(encoding="utf-8").splitlines()[:-1])
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "evaluate-judgments",
            "--cases-jsonl",
            str(FIXTURE),
            "--judgments-jsonl",
            str(incomplete),
            "--judgments-per-pair",
            "4",
            "--minimum-per-order",
            "2",
        ]
    )

    assert exit_code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["evaluation"] is None
    assert report["audit"]["count_mismatch_pair_ids"] == ["pref-test-gamma"]
