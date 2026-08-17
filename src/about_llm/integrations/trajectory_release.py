"""Fail-closed release checks for projected Agent trajectories.

The gate validates an authored, provider-neutral publication schema. It does
not sanitize raw provider responses or inspect opaque payload contents.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

TRAJECTORY_RELEASE_VERSION = "about-llm.trajectory-release.v1"

FindingCategory = Literal[
    "forbidden_field",
    "forbidden_block_type",
    "invalid_schema",
    "unknown_block_type",
]

_TOP_LEVEL_FIELDS = frozenset({"schema_version", "trajectory_id", "turns"})
_TURN_FIELDS = frozenset({"turn_id", "role", "blocks"})
_ROLES = frozenset({"system", "user", "assistant", "tool"})
_BLOCK_FIELDS: Mapping[str, frozenset[str]] = {
    "text": frozenset({"type", "text"}),
    "tool_call": frozenset({"type", "call_id", "tool_name", "arguments"}),
    "tool_result": frozenset({"type", "call_id", "status", "content"}),
    "citation": frozenset({"type", "source_id", "quote"}),
}
_FORBIDDEN_TERMS = frozenset(
    {
        "analysis",
        "encryptedcontent",
        "encryptedreasoning",
        "opaque",
        "reasoning",
        "reasoningcontent",
        "reasoningdetails",
        "redactedthinking",
        "signature",
        "thoughtsignature",
        "thinking",
        "thinkingsignature",
    }
)


@dataclass(frozen=True)
class TrajectoryReleaseFinding:
    """Value-redacted reason that a projected trajectory cannot be released."""

    path: str
    category: FindingCategory
    name: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "category": self.category, "name": self.name}


def build_trajectory_release_report(
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate strict release projections without echoing their values."""
    if isinstance(trajectories, (str, bytes)):
        raise ValueError("trajectories must be a sequence of JSON objects")
    findings: list[TrajectoryReleaseFinding] = []
    for index, trajectory in enumerate(trajectories):
        _audit_trajectory(trajectory, path=f"$[{index}]", findings=findings)
    opaque_count = sum(
        finding.category in {"forbidden_field", "forbidden_block_type"}
        for finding in findings
    )
    unknown_count = sum(
        finding.category == "unknown_block_type" for finding in findings
    )
    return {
        "passed": not findings,
        "schema_version": TRAJECTORY_RELEASE_VERSION,
        "network_performed": False,
        "provider_artifacts_interpreted": False,
        "plaintext_values_emitted": False,
        "secret_pii_scan_performed": False,
        "trajectory_count": len(trajectories),
        "opaque_reasoning_block_count": opaque_count,
        "unknown_block_count": unknown_count,
        "finding_count": len(findings),
        "findings": [finding.as_dict() for finding in findings],
        "evidence_boundary": (
            "This gate checks a strict authored projection only; it does not sanitize raw "
            "provider responses, decode opaque blocks, or detect all secrets and PII."
        ),
    }


def _audit_trajectory(
    trajectory: Mapping[str, Any],
    *,
    path: str,
    findings: list[TrajectoryReleaseFinding],
) -> None:
    if not isinstance(trajectory, dict):
        _add(findings, path, "invalid_schema", "trajectory_not_object")
        return
    _audit_exact_fields(trajectory, _TOP_LEVEL_FIELDS, path=path, findings=findings)
    if trajectory.get("schema_version") != TRAJECTORY_RELEASE_VERSION:
        _add(findings, f"{path}.schema_version", "invalid_schema", "unsupported_version")
    _audit_non_empty_string(trajectory.get("trajectory_id"), f"{path}.trajectory_id", findings)
    turns = trajectory.get("turns")
    if not isinstance(turns, list) or not turns:
        _add(findings, f"{path}.turns", "invalid_schema", "non_empty_array_required")
        return
    for index, turn in enumerate(turns):
        _audit_turn(turn, path=f"{path}.turns[{index}]", findings=findings)


def _audit_turn(
    turn: Any,
    *,
    path: str,
    findings: list[TrajectoryReleaseFinding],
) -> None:
    if not isinstance(turn, dict):
        _add(findings, path, "invalid_schema", "turn_not_object")
        return
    _audit_exact_fields(turn, _TURN_FIELDS, path=path, findings=findings)
    _audit_non_empty_string(turn.get("turn_id"), f"{path}.turn_id", findings)
    if turn.get("role") not in _ROLES:
        _add(findings, f"{path}.role", "invalid_schema", "unsupported_role")
    blocks = turn.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        _add(findings, f"{path}.blocks", "invalid_schema", "non_empty_array_required")
        return
    for index, block in enumerate(blocks):
        _audit_block(block, path=f"{path}.blocks[{index}]", findings=findings)


def _audit_block(
    block: Any,
    *,
    path: str,
    findings: list[TrajectoryReleaseFinding],
) -> None:
    if not isinstance(block, dict):
        _add(findings, path, "invalid_schema", "block_not_object")
        return
    block_type = block.get("type")
    if not isinstance(block_type, str) or not block_type:
        _add(findings, f"{path}.type", "invalid_schema", "block_type_required")
        return
    normalized_type = _normalize_name(block_type)
    if normalized_type in _FORBIDDEN_TERMS:
        _add(findings, f"{path}.type", "forbidden_block_type", normalized_type)
        return
    expected_fields = _BLOCK_FIELDS.get(block_type)
    if expected_fields is None:
        _add(findings, f"{path}.type", "unknown_block_type", "unrecognized")
        return
    _audit_exact_fields(block, expected_fields, path=path, findings=findings)
    for name in block:
        normalized_name = _normalize_name(name)
        if normalized_name in _FORBIDDEN_TERMS:
            _add(findings, f"{path}.{name}", "forbidden_field", normalized_name)
    if block_type == "text":
        _audit_string(block.get("text"), f"{path}.text", findings)
    elif block_type == "tool_call":
        _audit_non_empty_string(block.get("call_id"), f"{path}.call_id", findings)
        _audit_non_empty_string(block.get("tool_name"), f"{path}.tool_name", findings)
        arguments = block.get("arguments")
        if not isinstance(arguments, dict):
            _add(findings, f"{path}.arguments", "invalid_schema", "object_required")
        else:
            _audit_forbidden_fields(
                arguments,
                path=f"{path}.arguments",
                findings=findings,
            )
    elif block_type == "tool_result":
        _audit_non_empty_string(block.get("call_id"), f"{path}.call_id", findings)
        if block.get("status") not in {"ok", "error"}:
            _add(findings, f"{path}.status", "invalid_schema", "unsupported_status")
        _audit_string(block.get("content"), f"{path}.content", findings)
    else:
        _audit_non_empty_string(block.get("source_id"), f"{path}.source_id", findings)
        _audit_string(block.get("quote"), f"{path}.quote", findings)


def _audit_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    path: str,
    findings: list[TrajectoryReleaseFinding],
) -> None:
    actual = set(value)
    for name in sorted(actual - expected):
        normalized_name = _normalize_name(name)
        if normalized_name in _FORBIDDEN_TERMS:
            _add(
                findings,
                f"{path}.<forbidden-field>",
                "forbidden_field",
                normalized_name,
            )
        else:
            _add(
                findings,
                f"{path}.<unknown-field>",
                "invalid_schema",
                "unknown_field",
            )
    for name in sorted(expected - actual):
        _add(findings, f"{path}.{name}", "invalid_schema", "missing_field")


def _audit_non_empty_string(
    value: Any,
    path: str,
    findings: list[TrajectoryReleaseFinding],
) -> None:
    if not isinstance(value, str) or not value:
        _add(findings, path, "invalid_schema", "non_empty_string_required")


def _audit_string(
    value: Any,
    path: str,
    findings: list[TrajectoryReleaseFinding],
) -> None:
    if not isinstance(value, str):
        _add(findings, path, "invalid_schema", "string_required")


def _audit_forbidden_fields(
    value: Any,
    *,
    path: str,
    findings: list[TrajectoryReleaseFinding],
) -> None:
    if isinstance(value, dict):
        for name, child in value.items():
            normalized_name = _normalize_name(name)
            if normalized_name in _FORBIDDEN_TERMS:
                _add(
                    findings,
                    f"{path}.<forbidden-field>",
                    "forbidden_field",
                    normalized_name,
                )
            _audit_forbidden_fields(child, path=f"{path}.*", findings=findings)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _audit_forbidden_fields(
                child,
                path=f"{path}[{index}]",
                findings=findings,
            )


def _add(
    findings: list[TrajectoryReleaseFinding],
    path: str,
    category: FindingCategory,
    name: str,
) -> None:
    findings.append(TrajectoryReleaseFinding(path, category, name))


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())
