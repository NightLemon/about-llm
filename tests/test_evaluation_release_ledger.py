from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from about_llm.evaluation import (
    EVALUATION_RELEASE_LEDGER_EVIDENCE_BOUNDARY,
    EvaluationReleaseLedger,
    append_evaluation_release_artifact,
    append_evaluation_release_record,
    load_evaluation_release_ledger,
    verify_evaluation_release_ledger,
    write_evaluation_release_ledger,
)
from about_llm.llmops import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "evaluation-gate"
FIXTURE_KEYS = {
    # Public protocol fixtures, deliberately not production secrets.
    "fixture-hmac-2026-a": bytes.fromhex("11" * 32),
    "fixture-hmac-2026-b": bytes.fromhex("22" * 32),
}


def _ledger(
    tmp_path: Path,
) -> tuple[EvaluationReleaseLedger, dict[str, Path], dict[str, bytes]]:
    keys = {
        "release-key-a": bytes.fromhex("a1" * 32),
        "release-key-b": bytes.fromhex("b2" * 32),
    }
    paths = {
        "baseline-run": tmp_path / "baseline.json",
        "candidate-run": tmp_path / "candidate.json",
        "comparison": tmp_path / "comparison.json",
    }
    paths["baseline-run"].write_bytes(b'{"run":"baseline"}\n')
    paths["candidate-run"].write_bytes(b'{"run":"candidate"}\n')
    paths["comparison"].write_bytes(b'{"passed":true}\n')
    ledger = append_evaluation_release_artifact(
        None,
        release_id="release-001-baseline",
        artifact_id="baseline-run",
        artifact_kind="evaluation_run_manifest",
        artifact_path=paths["baseline-run"],
        decision="recorded",
        recorded_at="2026-08-07T09:00:00+08:00",
        key_id="release-key-a",
        secret_key=keys["release-key-a"],
    )
    ledger = append_evaluation_release_artifact(
        ledger,
        release_id="release-002-candidate",
        artifact_id="candidate-run",
        artifact_kind="evaluation_run_manifest",
        artifact_path=paths["candidate-run"],
        decision="recorded",
        recorded_at="2026-08-07T09:01:00+08:00",
        key_id="release-key-a",
        secret_key=keys["release-key-a"],
    )
    ledger = append_evaluation_release_artifact(
        ledger,
        release_id="release-003-gate",
        artifact_id="comparison",
        artifact_kind="evaluation_comparison",
        artifact_path=paths["comparison"],
        decision="approved",
        recorded_at="2026-08-07T09:02:00+08:00",
        key_id="release-key-b",
        secret_key=keys["release-key-b"],
    )
    return ledger, paths, keys


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def test_round_trip_verifies_key_rotation_artifacts_and_trusted_head(
    tmp_path: Path,
) -> None:
    ledger, paths, keys = _ledger(tmp_path)
    path = tmp_path / "release-ledger.json"
    write_evaluation_release_ledger(path, ledger)

    loaded = load_evaluation_release_ledger(path)
    verification = verify_evaluation_release_ledger(
        loaded,
        key_resolver=keys,
        artifact_paths=paths,
        trusted_head=ledger.head,
    )

    assert loaded == ledger
    assert [record.key_id for record in loaded.records] == [
        "release-key-a",
        "release-key-a",
        "release-key-b",
    ]
    assert verification.referenced_artifacts_rehashed is True
    assert verification.trusted_head_matched is True
    assert verification.to_dict()["authenticated_chain"] is True
    assert verification.to_dict()["timestamp_authority_verified"] is False
    assert "tail truncation" in EVALUATION_RELEASE_LEDGER_EVIDENCE_BOUNDARY


def test_project_fixture_authenticates_current_artifact_bytes() -> None:
    ledger = load_evaluation_release_ledger(PROJECT / "release-ledger.example.json")
    verification = verify_evaluation_release_ledger(
        ledger,
        key_resolver=FIXTURE_KEYS,
        artifact_paths={
            "baseline-run-manifest": PROJECT / "run.baseline.manifest.example.json",
            "candidate-run-manifest": PROJECT / "run.candidate.manifest.example.json",
            "release-comparison": PROJECT / "comparison.example.json",
        },
        trusted_head=ledger.head,
    )

    assert verification.record_count == 3
    assert ledger.records[-1].decision == "approved"
    assert ledger.records[-1].key_id == "fixture-hmac-2026-b"


def test_record_tampering_and_wrong_key_fail_authentication(tmp_path: Path) -> None:
    ledger, _, keys = _ledger(tmp_path)
    tampered_first = replace(ledger.records[0], release_id="rewritten-release")
    tampered = EvaluationReleaseLedger((tampered_first, *ledger.records[1:]))

    with pytest.raises(ValueError, match="record 1 MAC verification failed"):
        verify_evaluation_release_ledger(tampered, key_resolver=keys)
    with pytest.raises(ValueError, match="record 3 MAC verification failed"):
        verify_evaluation_release_ledger(
            ledger,
            key_resolver={**keys, "release-key-b": bytes.fromhex("cc" * 32)},
        )


def test_reordering_is_rejected_before_mac_verification(tmp_path: Path) -> None:
    ledger, _, _ = _ledger(tmp_path)
    payload = ledger.to_dict()
    records = payload["records"]
    assert isinstance(records, list)
    records[0], records[1] = records[1], records[0]
    path = tmp_path / "reordered.json"
    _write_canonical(path, payload)

    with pytest.raises(ValueError, match="sequence must be contiguous"):
        load_evaluation_release_ledger(path)


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "nonfinite", "pretty"])
def test_loader_requires_unambiguous_canonical_json(
    tmp_path: Path, mutation: str
) -> None:
    ledger, _, _ = _ledger(tmp_path)
    path = tmp_path / "invalid.json"
    if mutation == "duplicate":
        path.write_text('{"ledger_version":"a","ledger_version":"b"}\n', encoding="utf-8")
        message = "duplicate JSON object key"
    elif mutation == "nonfinite":
        path.write_text('{"ledger_version":NaN}\n', encoding="utf-8")
        message = "non-standard JSON constant"
    elif mutation == "pretty":
        path.write_text(json.dumps(ledger.to_dict(), indent=2) + "\n", encoding="utf-8")
        message = "not canonical JSON"
    else:
        payload = ledger.to_dict()
        payload["unexpected"] = True
        _write_canonical(path, payload)
        message = "unknown=\\['unexpected'\\]"

    with pytest.raises(ValueError, match=message):
        load_evaluation_release_ledger(path)


def test_exact_artifact_mapping_and_byte_drift_fail_closed(tmp_path: Path) -> None:
    ledger, paths, keys = _ledger(tmp_path)
    incomplete = {key: value for key, value in paths.items() if key != "comparison"}

    with pytest.raises(ValueError, match="exactly match ledger artifact ids"):
        verify_evaluation_release_ledger(
            ledger, key_resolver=keys, artifact_paths=incomplete
        )

    paths["comparison"].write_bytes(b'{"passed":nope}\n')
    with pytest.raises(ValueError, match=r"comparison.*SHA-256"):
        verify_evaluation_release_ledger(
            ledger, key_resolver=keys, artifact_paths=paths
        )


def test_trusted_head_is_required_to_detect_valid_prefix_truncation(
    tmp_path: Path,
) -> None:
    ledger, _, keys = _ledger(tmp_path)
    prefix = EvaluationReleaseLedger(ledger.records[:2])

    without_anchor = verify_evaluation_release_ledger(prefix, key_resolver=keys)
    assert without_anchor.trusted_head_matched is False
    assert without_anchor.referenced_artifacts_rehashed is False

    with pytest.raises(ValueError, match="externally trusted head"):
        verify_evaluation_release_ledger(
            prefix, key_resolver=keys, trusted_head=ledger.head
        )


def test_snapshot_write_refuses_overwrite(tmp_path: Path) -> None:
    ledger, _, _ = _ledger(tmp_path)
    path = tmp_path / "immutable.json"
    write_evaluation_release_ledger(path, ledger)

    with pytest.raises(FileExistsError):
        write_evaluation_release_ledger(path, ledger)


@pytest.mark.parametrize(
    ("key", "error", "message"),
    [
        (b"too-short", ValueError, "at least 32 bytes"),
        ("not-bytes", TypeError, "must be bytes"),
    ],
)
def test_secret_key_contract_is_explicit(
    key: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        append_evaluation_release_record(
            None,
            release_id="release",
            artifact_id="artifact",
            artifact_kind="comparison",
            artifact_bytes=b"{}\n",
            decision="approved",
            recorded_at="2026-08-07T00:00:00Z",
            key_id="key",
            secret_key=key,
        )


def test_missing_rotated_key_fails_closed(tmp_path: Path) -> None:
    ledger, _, keys = _ledger(tmp_path)

    with pytest.raises(ValueError, match="no verification key"):
        verify_evaluation_release_ledger(
            ledger,
            key_resolver={"release-key-a": keys["release-key-a"]},
        )
