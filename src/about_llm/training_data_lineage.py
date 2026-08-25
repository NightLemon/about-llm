"""Executable source-to-checkpoint lineage and deletion-impact analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from pathlib import Path
from typing import Any, NoReturn, cast

from about_llm.llmops import artifact_fingerprint

TRAINING_DATA_LINEAGE_VERSION = "about-llm.training-data-lineage.v1"
TRAINING_DATA_LINEAGE_REPORT_VERSION = "about-llm.training-data-lineage-report.v1"

_SPEC_FIELDS = {
    "artifacts",
    "case_id",
    "consumptions",
    "dedup_clusters",
    "deletion_request",
    "placements",
    "schema_version",
    "shard_spans",
    "source_snapshots",
    "trace_snapshot_id",
}
_SNAPSHOT_FIELDS = {
    "captured_at",
    "content_sha256",
    "registry_decision_id",
    "revision",
    "snapshot_id",
    "source_id",
    "use_allowed",
}
_ARTIFACT_FIELDS = {
    "artifact_id",
    "content_sha256",
    "parent_id",
    "stage",
    "transform_revision",
}
_CLUSTER_FIELDS = {"canonical_id", "cluster_id", "member_ids", "policy_revision"}
_PLACEMENT_FIELDS = {
    "dataset_version",
    "item_id",
    "mixture_component",
    "sampling_weight",
    "split",
}
_SPAN_FIELDS = {
    "dataset_version",
    "end_token",
    "item_id",
    "shard_id",
    "start_token",
}
_CONSUMPTION_FIELDS = {
    "checkpoint_ids",
    "dataset_version",
    "run_id",
    "shard_ids",
}
_DELETION_FIELDS = {"reason", "request_id", "requested_at", "scope", "source_id"}
_MAX_INPUT_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class SourceSnapshot:
    snapshot_id: str
    source_id: str
    revision: str
    content_sha256: str
    captured_at: str
    registry_decision_id: str
    use_allowed: bool


@dataclass(frozen=True)
class DerivedArtifact:
    artifact_id: str
    stage: str
    parent_id: str
    transform_revision: str
    content_sha256: str


@dataclass(frozen=True)
class DedupCluster:
    cluster_id: str
    canonical_id: str
    member_ids: tuple[str, ...]
    policy_revision: str


@dataclass(frozen=True)
class DatasetPlacement:
    item_id: str
    dataset_version: str
    split: str
    mixture_component: str
    sampling_weight: float


@dataclass(frozen=True)
class ShardSpan:
    item_id: str
    dataset_version: str
    shard_id: str
    start_token: int
    end_token: int


@dataclass(frozen=True)
class TrainingConsumption:
    run_id: str
    dataset_version: str
    shard_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]


@dataclass(frozen=True)
class DeletionRequest:
    request_id: str
    source_id: str
    scope: str
    requested_at: str
    reason: str


@dataclass(frozen=True)
class TrainingDataLineageSpec:
    case_id: str
    trace_snapshot_id: str
    source_snapshots: tuple[SourceSnapshot, ...]
    artifacts: tuple[DerivedArtifact, ...]
    dedup_clusters: tuple[DedupCluster, ...]
    placements: tuple[DatasetPlacement, ...]
    shard_spans: tuple[ShardSpan, ...]
    consumptions: tuple[TrainingConsumption, ...]
    deletion_request: DeletionRequest


def load_training_data_lineage_spec(path: Path) -> TrainingDataLineageSpec:
    """Load and validate one bounded, strict-JSON lineage example."""

    record = _load_json_object(path, "training data lineage spec")
    _exact_fields(record, _SPEC_FIELDS, "training data lineage spec")
    if record.get("schema_version") != TRAINING_DATA_LINEAGE_VERSION:
        raise ValueError("unsupported training data lineage schema_version")
    spec = TrainingDataLineageSpec(
        case_id=_string(record.get("case_id"), "case_id"),
        trace_snapshot_id=_string(
            record.get("trace_snapshot_id"), "trace_snapshot_id"
        ),
        source_snapshots=tuple(
            _parse_snapshot(item, index)
            for index, item in enumerate(
                _array(record.get("source_snapshots"), "source_snapshots")
            )
        ),
        artifacts=tuple(
            _parse_artifact(item, index)
            for index, item in enumerate(_array(record.get("artifacts"), "artifacts"))
        ),
        dedup_clusters=tuple(
            _parse_cluster(item, index)
            for index, item in enumerate(
                _array(record.get("dedup_clusters"), "dedup_clusters")
            )
        ),
        placements=tuple(
            _parse_placement(item, index)
            for index, item in enumerate(
                _array(record.get("placements"), "placements")
            )
        ),
        shard_spans=tuple(
            _parse_span(item, index)
            for index, item in enumerate(
                _array(record.get("shard_spans"), "shard_spans")
            )
        ),
        consumptions=tuple(
            _parse_consumption(item, index)
            for index, item in enumerate(
                _array(record.get("consumptions"), "consumptions")
            )
        ),
        deletion_request=_parse_deletion(record.get("deletion_request")),
    )
    _validate_spec(spec)
    return spec


def analyze_training_data_lineage(spec: TrainingDataLineageSpec) -> dict[str, object]:
    """Trace one snapshot and compute the concrete impact of one deletion request."""

    _validate_spec(spec)
    snapshots = {item.snapshot_id: item for item in spec.source_snapshots}
    artifacts = {item.artifact_id: item for item in spec.artifacts}
    traced = snapshots[spec.trace_snapshot_id]
    traced_descendants = _descendant_ids((traced.snapshot_id,), spec.artifacts)
    traced_parsed = [
        item.artifact_id
        for item in spec.artifacts
        if item.artifact_id in traced_descendants and item.stage == "parsed"
    ]
    traced_normalized = [
        item.artifact_id
        for item in spec.artifacts
        if item.artifact_id in traced_descendants and item.stage == "normalized"
    ]
    traced_normalized_set = set(traced_normalized)
    traced_clusters = [
        cluster
        for cluster in spec.dedup_clusters
        if set(cluster.member_ids) & traced_normalized_set
    ]
    traced_canonical_ids = {
        cluster.canonical_id
        for cluster in traced_clusters
        if cluster.canonical_id in traced_normalized
    }
    traced_placements = [
        item for item in spec.placements if item.item_id in traced_canonical_ids
    ]
    traced_spans = [
        item
        for item in spec.shard_spans
        if any(
            item.item_id == placement.item_id
            and item.dataset_version == placement.dataset_version
            for placement in traced_placements
        )
    ]
    traced_shards = {item.shard_id for item in traced_spans}
    traced_consumers = [
        item
        for item in spec.consumptions
        if traced_shards.intersection(item.shard_ids)
    ]

    deletion = spec.deletion_request
    affected_snapshots = [
        item.snapshot_id
        for item in spec.source_snapshots
        if item.source_id == deletion.source_id
    ]
    affected_artifacts = _descendant_ids(tuple(affected_snapshots), spec.artifacts)
    affected_normalized = {
        artifact_id
        for artifact_id in affected_artifacts
        if artifacts[artifact_id].stage == "normalized"
    }
    affected_clusters = [
        cluster
        for cluster in spec.dedup_clusters
        if affected_normalized.intersection(cluster.member_ids)
    ]
    removed_canonical_ids = {
        cluster.canonical_id
        for cluster in affected_clusters
        if cluster.canonical_id in affected_normalized
    }
    removed_placements = [
        item for item in spec.placements if item.item_id in removed_canonical_ids
    ]
    rebuild_spans = [
        item
        for item in spec.shard_spans
        if any(
            item.item_id == placement.item_id
            and item.dataset_version == placement.dataset_version
            for placement in removed_placements
        )
    ]
    rebuild_shards = {item.shard_id for item in rebuild_spans}
    impacted_consumptions = [
        item
        for item in spec.consumptions
        if rebuild_shards.intersection(item.shard_ids)
    ]
    cluster_actions = [
        {
            "cluster_id": cluster.cluster_id,
            "removed_member_ids": [
                item for item in cluster.member_ids if item in affected_normalized
            ],
            "removed_canonical_id": (
                cluster.canonical_id
                if cluster.canonical_id in affected_normalized
                else None
            ),
            "surviving_member_ids": [
                item for item in cluster.member_ids if item not in affected_normalized
            ],
            "next_action": (
                "review-surviving-members"
                if any(item not in affected_normalized for item in cluster.member_ids)
                else "remove-empty-cluster"
            ),
            "automatic_canonical_replacement_allowed": False,
        }
        for cluster in affected_clusters
    ]

    report: dict[str, object] = {
        "report_version": TRAINING_DATA_LINEAGE_REPORT_VERSION,
        "case_id": spec.case_id,
        "trace": {
            "stable_source_id": traced.source_id,
            "known_revisions": [
                {
                    "snapshot_id": item.snapshot_id,
                    "revision": item.revision,
                    "content_sha256": item.content_sha256,
                }
                for item in spec.source_snapshots
                if item.source_id == traced.source_id
            ],
            "traced_snapshot_id": traced.snapshot_id,
            "parsed_artifact_ids": traced_parsed,
            "normalized_artifact_ids": traced_normalized,
            "dedup_clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "canonical_id": cluster.canonical_id,
                    "member_ids": list(cluster.member_ids),
                    "traced_source_is_canonical": (
                        cluster.canonical_id in traced_normalized
                    ),
                }
                for cluster in traced_clusters
            ],
            "placements": [
                {
                    "item_id": item.item_id,
                    "dataset_version": item.dataset_version,
                    "split": item.split,
                    "mixture_component": item.mixture_component,
                    "sampling_weight": item.sampling_weight,
                }
                for item in traced_placements
            ],
            "shard_spans": [
                {
                    "item_id": item.item_id,
                    "dataset_version": item.dataset_version,
                    "shard_id": item.shard_id,
                    "start_token": item.start_token,
                    "end_token": item.end_token,
                    "objective_token_count": item.end_token - item.start_token,
                }
                for item in traced_spans
            ],
            "consumers": [
                {
                    "run_id": item.run_id,
                    "dataset_version": item.dataset_version,
                    "shard_ids": list(item.shard_ids),
                    "checkpoint_ids": list(item.checkpoint_ids),
                }
                for item in traced_consumers
            ],
        },
        "deletion_impact": {
            "request_id": deletion.request_id,
            "source_id": deletion.source_id,
            "scope": deletion.scope,
            "requested_at": deletion.requested_at,
            "reason": deletion.reason,
            "future_build_tombstone_required": True,
            "affected_snapshot_ids": affected_snapshots,
            "affected_artifact_ids": sorted(affected_artifacts),
            "cluster_actions": cluster_actions,
            "removed_dataset_items": [item.item_id for item in removed_placements],
            "dataset_versions_requiring_rebuild": sorted(
                {item.dataset_version for item in removed_placements}
            ),
            "shards_requiring_rebuild": sorted(rebuild_shards),
            "impacted_runs": [item.run_id for item in impacted_consumptions],
            "impacted_checkpoints": [
                checkpoint
                for item in impacted_consumptions
                for checkpoint in item.checkpoint_ids
            ],
            "stored_data_deleted_by_this_analysis": False,
            "trained_weight_effect_removed_by_shard_rebuild": False,
        },
        "invariants": {
            "stable_source_id_can_have_multiple_revisions": len(
                {
                    item.revision
                    for item in spec.source_snapshots
                    if item.source_id == traced.source_id
                }
            )
            > 1,
            "source_revision_pairs_are_unique": _values_are_unique(
                (item.source_id, item.revision) for item in spec.source_snapshots
            ),
            "normalized_items_have_one_dedup_cluster": True,
            "placed_items_are_cluster_canonicals": True,
            "shard_spans_do_not_overlap": True,
            "consumed_shards_are_declared": True,
        },
        "evidence_boundary": (
            "This deterministic example validates one declared lineage graph and computes "
            "which manifests, shards, runs, and checkpoints are linked to one source-level "
            "deletion request. It does not execute deletion, authenticate source records, "
            "decide whether a mirror may be reused, remove information from trained weights, "
            "or establish legal compliance."
        ),
    }
    report["report_fingerprint"] = "sha256:" + artifact_fingerprint(report)
    return report


def verify_training_data_lineage_report(
    spec: TrainingDataLineageSpec, report_path: Path
) -> dict[str, object]:
    """Recompute the deterministic report and reject any recorded drift."""

    observed = _load_json_object(report_path, "training data lineage report")
    expected = analyze_training_data_lineage(spec)
    if observed != expected:
        raise ValueError("recorded training data lineage report differs from recomputation")
    return cast(dict[str, object], observed)


def _validate_spec(spec: TrainingDataLineageSpec) -> None:
    if not isinstance(spec, TrainingDataLineageSpec):
        raise ValueError("spec must be TrainingDataLineageSpec")
    _unique((item.snapshot_id for item in spec.source_snapshots), "snapshot_id")
    _unique(
        (
            (item.source_id, item.revision)
            for item in spec.source_snapshots
        ),
        "source_id/revision",
    )
    _unique((item.artifact_id for item in spec.artifacts), "artifact_id")
    _unique((item.cluster_id for item in spec.dedup_clusters), "cluster_id")
    _unique((item.run_id for item in spec.consumptions), "run_id")
    snapshots = {item.snapshot_id: item for item in spec.source_snapshots}
    artifacts = {item.artifact_id: item for item in spec.artifacts}
    if spec.trace_snapshot_id not in snapshots:
        raise ValueError("trace_snapshot_id does not name a source snapshot")
    if not spec.source_snapshots or not spec.artifacts or not spec.dedup_clusters:
        raise ValueError("lineage example requires snapshots, artifacts, and clusters")
    for snapshot in spec.source_snapshots:
        if not snapshot.use_allowed:
            raise ValueError("this teaching build accepts only registry-approved snapshots")
    for artifact in spec.artifacts:
        if artifact.stage == "parsed":
            if artifact.parent_id not in snapshots:
                raise ValueError("parsed artifact parent must be a source snapshot")
        elif artifact.stage == "normalized":
            parent = artifacts.get(artifact.parent_id)
            if parent is None or parent.stage != "parsed":
                raise ValueError("normalized artifact parent must be parsed")
        else:
            raise ValueError("artifact stage must be parsed or normalized")
    normalized = {
        item.artifact_id for item in spec.artifacts if item.stage == "normalized"
    }
    cluster_members: list[str] = []
    cluster_canonicals: set[str] = set()
    for cluster in spec.dedup_clusters:
        if not cluster.member_ids or len(set(cluster.member_ids)) != len(
            cluster.member_ids
        ):
            raise ValueError("dedup cluster members must be unique and non-empty")
        if not set(cluster.member_ids).issubset(normalized):
            raise ValueError("dedup cluster member must be a normalized artifact")
        if cluster.canonical_id not in cluster.member_ids:
            raise ValueError("dedup canonical must be a cluster member")
        cluster_members.extend(cluster.member_ids)
        cluster_canonicals.add(cluster.canonical_id)
    if set(cluster_members) != normalized or len(cluster_members) != len(normalized):
        raise ValueError("every normalized artifact must belong to exactly one cluster")
    placement_keys: set[tuple[str, str]] = set()
    for placement in spec.placements:
        if placement.item_id not in cluster_canonicals:
            raise ValueError("placed item must be a dedup cluster canonical")
        key = (placement.dataset_version, placement.item_id)
        if key in placement_keys:
            raise ValueError("dataset placement must be unique")
        placement_keys.add(key)
    span_keys: set[tuple[str, str]] = set()
    spans_by_shard: dict[tuple[str, str], list[ShardSpan]] = {}
    for span in spec.shard_spans:
        key = (span.dataset_version, span.item_id)
        if key not in placement_keys:
            raise ValueError("shard span must reference a placed dataset item")
        if key in span_keys:
            raise ValueError("this example permits one shard span per placed item")
        span_keys.add(key)
        spans_by_shard.setdefault((span.dataset_version, span.shard_id), []).append(
            span
        )
    if span_keys != placement_keys:
        raise ValueError("every placed item must have one shard span")
    for spans in spans_by_shard.values():
        ordered = sorted(spans, key=lambda item: item.start_token)
        if any(left.end_token > right.start_token for left, right in pairwise(ordered)):
            raise ValueError("shard token spans must not overlap")
    declared_shards = set(spans_by_shard)
    for consumption in spec.consumptions:
        if not consumption.shard_ids or not consumption.checkpoint_ids:
            raise ValueError("consumption must name shards and checkpoints")
        if len(set(consumption.shard_ids)) != len(consumption.shard_ids):
            raise ValueError("consumed shard ids must be unique")
        if len(set(consumption.checkpoint_ids)) != len(consumption.checkpoint_ids):
            raise ValueError("checkpoint ids must be unique within a run")
        for shard in consumption.shard_ids:
            if (consumption.dataset_version, shard) not in declared_shards:
                raise ValueError("consumption references an undeclared dataset shard")
    if spec.deletion_request.source_id not in {
        item.source_id for item in spec.source_snapshots
    }:
        raise ValueError("deletion request source_id is unknown")
    if spec.deletion_request.scope != "all_revisions":
        raise ValueError("this example requires an all_revisions deletion request")


def _descendant_ids(
    root_ids: tuple[str, ...], artifacts: tuple[DerivedArtifact, ...]
) -> set[str]:
    descendants: set[str] = set()
    parents = set(root_ids)
    changed = True
    while changed:
        changed = False
        for artifact in artifacts:
            if artifact.artifact_id not in descendants and artifact.parent_id in parents:
                descendants.add(artifact.artifact_id)
                parents.add(artifact.artifact_id)
                changed = True
    return descendants


def _parse_snapshot(value: object, index: int) -> SourceSnapshot:
    record = _object(value, f"source_snapshots[{index}]")
    _exact_fields(record, _SNAPSHOT_FIELDS, f"source_snapshots[{index}]")
    allowed = record.get("use_allowed")
    if not isinstance(allowed, bool):
        raise ValueError(f"source_snapshots[{index}].use_allowed must be boolean")
    return SourceSnapshot(
        snapshot_id=_string(record.get("snapshot_id"), "snapshot_id"),
        source_id=_string(record.get("source_id"), "source_id"),
        revision=_string(record.get("revision"), "revision"),
        content_sha256=_digest(record.get("content_sha256"), "content_sha256"),
        captured_at=_string(record.get("captured_at"), "captured_at"),
        registry_decision_id=_string(
            record.get("registry_decision_id"), "registry_decision_id"
        ),
        use_allowed=allowed,
    )


def _parse_artifact(value: object, index: int) -> DerivedArtifact:
    record = _object(value, f"artifacts[{index}]")
    _exact_fields(record, _ARTIFACT_FIELDS, f"artifacts[{index}]")
    return DerivedArtifact(
        artifact_id=_string(record.get("artifact_id"), "artifact_id"),
        stage=_string(record.get("stage"), "stage"),
        parent_id=_string(record.get("parent_id"), "parent_id"),
        transform_revision=_string(
            record.get("transform_revision"), "transform_revision"
        ),
        content_sha256=_digest(record.get("content_sha256"), "content_sha256"),
    )


def _parse_cluster(value: object, index: int) -> DedupCluster:
    record = _object(value, f"dedup_clusters[{index}]")
    _exact_fields(record, _CLUSTER_FIELDS, f"dedup_clusters[{index}]")
    return DedupCluster(
        cluster_id=_string(record.get("cluster_id"), "cluster_id"),
        canonical_id=_string(record.get("canonical_id"), "canonical_id"),
        member_ids=tuple(
            _string(item, "member_id")
            for item in _array(record.get("member_ids"), "member_ids")
        ),
        policy_revision=_string(
            record.get("policy_revision"), "policy_revision"
        ),
    )


def _parse_placement(value: object, index: int) -> DatasetPlacement:
    record = _object(value, f"placements[{index}]")
    _exact_fields(record, _PLACEMENT_FIELDS, f"placements[{index}]")
    weight = record.get("sampling_weight")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not isfinite(weight)
        or weight <= 0
    ):
        raise ValueError("sampling_weight must be a positive finite number")
    exact_weight = float(weight)
    split = _string(record.get("split"), "split")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split is unsupported")
    return DatasetPlacement(
        item_id=_string(record.get("item_id"), "item_id"),
        dataset_version=_string(record.get("dataset_version"), "dataset_version"),
        split=split,
        mixture_component=_string(
            record.get("mixture_component"), "mixture_component"
        ),
        sampling_weight=exact_weight,
    )


def _parse_span(value: object, index: int) -> ShardSpan:
    record = _object(value, f"shard_spans[{index}]")
    _exact_fields(record, _SPAN_FIELDS, f"shard_spans[{index}]")
    start = _nonnegative_integer(record.get("start_token"), "start_token")
    end = _positive_integer(record.get("end_token"), "end_token")
    if end <= start:
        raise ValueError("shard span end_token must be greater than start_token")
    return ShardSpan(
        item_id=_string(record.get("item_id"), "item_id"),
        dataset_version=_string(record.get("dataset_version"), "dataset_version"),
        shard_id=_string(record.get("shard_id"), "shard_id"),
        start_token=start,
        end_token=end,
    )


def _parse_consumption(value: object, index: int) -> TrainingConsumption:
    record = _object(value, f"consumptions[{index}]")
    _exact_fields(record, _CONSUMPTION_FIELDS, f"consumptions[{index}]")
    return TrainingConsumption(
        run_id=_string(record.get("run_id"), "run_id"),
        dataset_version=_string(record.get("dataset_version"), "dataset_version"),
        shard_ids=tuple(
            _string(item, "shard_id")
            for item in _array(record.get("shard_ids"), "shard_ids")
        ),
        checkpoint_ids=tuple(
            _string(item, "checkpoint_id")
            for item in _array(record.get("checkpoint_ids"), "checkpoint_ids")
        ),
    )


def _parse_deletion(value: object) -> DeletionRequest:
    record = _object(value, "deletion_request")
    _exact_fields(record, _DELETION_FIELDS, "deletion_request")
    return DeletionRequest(
        request_id=_string(record.get("request_id"), "request_id"),
        source_id=_string(record.get("source_id"), "source_id"),
        scope=_string(record.get("scope"), "scope"),
        requested_at=_string(record.get("requested_at"), "requested_at"),
        reason=_string(record.get("reason"), "reason"),
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    if path.stat().st_size > _MAX_INPUT_BYTES:
        raise ValueError(f"{label} exceeds the input byte limit")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be UTF-8 strict JSON") from error
    return _object(value, label)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique(values: Any, label: str) -> None:
    snapshot = tuple(values)
    if not snapshot or not _values_are_unique(snapshot):
        raise ValueError(f"{label} values must be unique and non-empty")


def _values_are_unique(values: Any) -> bool:
    snapshot = tuple(values)
    return len(snapshot) == len(set(snapshot))


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    return value


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _exact_fields(record: dict[str, Any], expected: set[str], label: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{label} fields differ from the supported schema")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} contains an unpaired Unicode surrogate") from error
    return value


def _digest(value: object, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise ValueError(f"{label} must be lowercase SHA-256")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise ValueError(f"{label} must be lowercase SHA-256") from error
    if digest != digest.lower():
        raise ValueError(f"{label} must be lowercase SHA-256")
    return digest


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: object, label: str) -> int:
    result = _nonnegative_integer(value, label)
    if result == 0:
        raise ValueError(f"{label} must be a positive integer")
    return result
