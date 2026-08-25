from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from about_llm.training_data_lineage import (
    analyze_training_data_lineage,
    load_training_data_lineage_spec,
    verify_training_data_lineage_report,
)

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "training-data-lineage"
SPEC = PROJECT / "thread-8841.lineage.json"
REPORT = PROJECT / "thread-8841.recorded-report.json"
SCRIPT = PROJECT / "thread_lineage.py"


def _payload() -> dict[str, Any]:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def _write_changed_spec(
    tmp_path: Path, change: Callable[[dict[str, Any]], None]
) -> Path:
    payload = _payload()
    change(payload)
    path = tmp_path / "lineage.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path


def test_thread_trace_reaches_exact_token_spans_and_checkpoints() -> None:
    report = analyze_training_data_lineage(load_training_data_lineage_spec(SPEC))

    trace = report["trace"]
    assert isinstance(trace, dict)
    assert [item["revision"] for item in trace["known_revisions"]] == ["r2", "r3"]
    assert [item["objective_token_count"] for item in trace["shard_spans"]] == [
        96,
        64,
    ]
    assert {item["shard_id"] for item in trace["shard_spans"]} == {
        "train-00031.bin"
    }
    assert trace["consumers"] == [
        {
            "run_id": "pretrain-cn-run-42",
            "dataset_version": "pretrain-cn-2026-08-v3",
            "shard_ids": ["train-00031.bin"],
            "checkpoint_ids": [
                "pretrain-cn-run-42/step-12000",
                "pretrain-cn-run-42/step-18000",
            ],
        }
    ]


def test_all_revision_deletion_marks_rebuild_without_claiming_unlearning() -> None:
    report = analyze_training_data_lineage(load_training_data_lineage_spec(SPEC))

    impact = report["deletion_impact"]
    assert isinstance(impact, dict)
    assert impact["requested_at"] == "2026-08-25T09:00:00Z"
    assert impact["reason"] == "source-owner-request"
    assert impact["affected_snapshot_ids"] == [
        "forum-cn/thread-8841@r2",
        "forum-cn/thread-8841@r3",
    ]
    assert impact["shards_requiring_rebuild"] == ["train-00031.bin"]
    assert impact["impacted_runs"] == ["pretrain-cn-run-42"]
    assert impact["stored_data_deleted_by_this_analysis"] is False
    assert impact["trained_weight_effect_removed_by_shard_rebuild"] is False


def test_deleting_canonical_requires_review_before_reusing_mirror() -> None:
    report = analyze_training_data_lineage(load_training_data_lineage_spec(SPEC))
    impact = report["deletion_impact"]
    assert isinstance(impact, dict)
    action = next(
        item
        for item in impact["cluster_actions"]
        if item["cluster_id"] == "thread-8841-main-r3"
    )

    assert action["surviving_member_ids"] == [
        "normalized/mirror-tech/thread-8841-copy/r1/main"
    ]
    assert action["next_action"] == "review-surviving-members"
    assert action["automatic_canonical_replacement_allowed"] is False


def test_recorded_report_is_recomputed_from_the_spec() -> None:
    report = verify_training_data_lineage_report(
        load_training_data_lineage_spec(SPEC), REPORT
    )

    assert report["report_fingerprint"] == (
        "sha256:acaf7c1ce7b599352c697c74c42a1bc3"
        "252f0e011ff98ed0171f46536f33d5a7"
    )


def test_same_source_revision_cannot_name_two_snapshots(tmp_path: Path) -> None:
    def duplicate_revision(payload: dict[str, Any]) -> None:
        duplicate = dict(payload["source_snapshots"][0])
        duplicate["snapshot_id"] = "forum-cn/thread-8841@r2-copy"
        payload["source_snapshots"].append(duplicate)

    path = _write_changed_spec(tmp_path, duplicate_revision)

    with pytest.raises(ValueError, match="source_id/revision"):
        load_training_data_lineage_spec(path)


def test_only_a_cluster_canonical_can_enter_a_dataset(tmp_path: Path) -> None:
    def place_mirror(payload: dict[str, Any]) -> None:
        payload["placements"][0]["item_id"] = (
            "normalized/mirror-tech/thread-8841-copy/r1/main"
        )

    path = _write_changed_spec(tmp_path, place_mirror)

    with pytest.raises(ValueError, match="dedup cluster canonical"):
        load_training_data_lineage_spec(path)


def test_token_spans_in_one_shard_cannot_overlap(tmp_path: Path) -> None:
    def overlap(payload: dict[str, Any]) -> None:
        payload["shard_spans"][1]["start_token"] = 4191

    path = _write_changed_spec(tmp_path, overlap)

    with pytest.raises(ValueError, match="must not overlap"):
        load_training_data_lineage_spec(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda text: text.replace(
                '"case_id": "thread-8841",',
                '"case_id": "thread-8841", "case_id": "changed",',
            ),
            "duplicate JSON key",
        ),
        (
            lambda text: text.replace(
                '"sampling_weight": 1.0', '"sampling_weight": NaN', 1
            ),
            "non-finite",
        ),
        (
            lambda text: text.replace(
                '"trace_snapshot_id":', '"unexpected": true, "trace_snapshot_id":', 1
            ),
            "fields differ",
        ),
        (
            lambda text: text.replace(
                '"source-owner-request"', '"\\ud800"', 1
            ),
            "unpaired Unicode surrogate",
        ),
    ],
)
def test_loader_rejects_ambiguous_or_unsupported_json(
    tmp_path: Path, change: Callable[[str], str], message: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(change(SPEC.read_text(encoding="utf-8")), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_training_data_lineage_spec(path)


@pytest.mark.integration
@pytest.mark.smoke
def test_project_cli_verifies_the_recorded_report() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "verify"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    payload = json.loads(completed.stdout)
    assert payload["verified"] is True
    assert payload["verification_scope"] == "full_local_recomputation"
