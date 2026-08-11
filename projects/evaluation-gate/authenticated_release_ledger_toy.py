from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.evaluation import (
    EvaluationReleaseLedger,
    append_evaluation_release_artifact,
    load_evaluation_release_ledger,
    verify_evaluation_release_ledger,
    write_evaluation_release_ledger,
)

PROJECT = Path(__file__).resolve().parent

# These values are public protocol fixtures. They are deliberately unsuitable
# for production authentication and prove nothing about secret-key custody.
FIXTURE_KEYS = {
    "fixture-hmac-2026-a": bytes.fromhex("11" * 32),
    "fixture-hmac-2026-b": bytes.fromhex("22" * 32),
}
ARTIFACT_PATHS = {
    "baseline-run-manifest": PROJECT / "run.baseline.manifest.example.json",
    "candidate-run-manifest": PROJECT / "run.candidate.manifest.example.json",
    "release-comparison": PROJECT / "comparison.example.json",
}


def build_fixture() -> EvaluationReleaseLedger:
    ledger = append_evaluation_release_artifact(
        None,
        release_id="authored-eval-release-001",
        artifact_id="baseline-run-manifest",
        artifact_kind="evaluation_run_manifest",
        artifact_path=ARTIFACT_PATHS["baseline-run-manifest"],
        decision="recorded",
        recorded_at="2026-08-07T09:00:00+08:00",
        key_id="fixture-hmac-2026-a",
        secret_key=FIXTURE_KEYS["fixture-hmac-2026-a"],
    )
    ledger = append_evaluation_release_artifact(
        ledger,
        release_id="authored-eval-release-002",
        artifact_id="candidate-run-manifest",
        artifact_kind="evaluation_run_manifest",
        artifact_path=ARTIFACT_PATHS["candidate-run-manifest"],
        decision="recorded",
        recorded_at="2026-08-07T09:01:00+08:00",
        key_id="fixture-hmac-2026-a",
        secret_key=FIXTURE_KEYS["fixture-hmac-2026-a"],
    )
    return append_evaluation_release_artifact(
        ledger,
        release_id="authored-eval-release-003",
        artifact_id="release-comparison",
        artifact_kind="evaluation_comparison",
        artifact_path=ARTIFACT_PATHS["release-comparison"],
        decision="approved",
        recorded_at="2026-08-07T09:02:00+08:00",
        key_id="fixture-hmac-2026-b",
        secret_key=FIXTURE_KEYS["fixture-hmac-2026-b"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the public authenticated release-ledger fixture"
    )
    parser.add_argument(
        "--write",
        type=Path,
        help="exclusive-create a canonical snapshot; refuses to overwrite",
    )
    args = parser.parse_args()
    expected = build_fixture()
    if args.write is not None:
        write_evaluation_release_ledger(args.write, expected)
        ledger = expected
    else:
        ledger = load_evaluation_release_ledger(
            PROJECT / "release-ledger.example.json"
        )
        if ledger != expected:
            raise ValueError("checked-in ledger does not match rebuilt artifact bytes")
    verification = verify_evaluation_release_ledger(
        ledger,
        key_resolver=FIXTURE_KEYS,
        artifact_paths=ARTIFACT_PATHS,
        trusted_head=expected.head,
    )
    payload = verification.to_dict()
    payload["fixture_keys_are_public_test_values"] = True
    payload["production_key_custody_proven"] = False
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
