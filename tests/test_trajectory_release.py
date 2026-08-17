from __future__ import annotations

import json

from about_llm.integrations.trajectory_release import (
    TRAJECTORY_RELEASE_VERSION,
    build_trajectory_release_report,
)


def safe_trajectory() -> dict[str, object]:
    return {
        "schema_version": TRAJECTORY_RELEASE_VERSION,
        "trajectory_id": "fixture-trajectory-001",
        "turns": [
            {
                "turn_id": "turn-001",
                "role": "user",
                "blocks": [{"type": "text", "text": "authored visible prompt"}],
            },
            {
                "turn_id": "turn-002",
                "role": "assistant",
                "blocks": [
                    {
                        "type": "tool_call",
                        "call_id": "call-001",
                        "tool_name": "lookup",
                        "arguments": {"query": "authored query"},
                    },
                    {
                        "type": "citation",
                        "source_id": "source-001",
                        "quote": "authored visible quote",
                    },
                ],
            },
            {
                "turn_id": "turn-003",
                "role": "tool",
                "blocks": [
                    {
                        "type": "tool_result",
                        "call_id": "call-001",
                        "status": "ok",
                        "content": "authored visible result",
                    }
                ],
            },
        ],
    }


def test_release_gate_accepts_only_the_strict_projection() -> None:
    report = build_trajectory_release_report([safe_trajectory()])

    assert report["passed"] is True
    assert report["trajectory_count"] == 1
    assert report["finding_count"] == 0
    assert report["opaque_reasoning_block_count"] == 0
    assert report["unknown_block_count"] == 0
    assert report["network_performed"] is False
    assert report["provider_artifacts_interpreted"] is False
    assert report["secret_pii_scan_performed"] is False


def test_release_gate_rejects_forbidden_and_unknown_blocks_without_values() -> None:
    trajectory = safe_trajectory()
    turns = trajectory["turns"]
    assert isinstance(turns, list)
    assistant = turns[1]
    assert isinstance(assistant, dict)
    blocks = assistant["blocks"]
    assert isinstance(blocks, list)
    blocks.extend(
        [
            {"type": "thinking", "thinking": "never-render-this-secret"},
            {"type": "future_provider_blob", "data": "never-render-this-payload"},
        ]
    )

    report = build_trajectory_release_report([trajectory])
    rendered = json.dumps(report, ensure_ascii=False)

    assert report["passed"] is False
    assert report["opaque_reasoning_block_count"] == 1
    assert report["unknown_block_count"] == 1
    assert report["finding_count"] == 2
    assert "forbidden_block_type" in rendered
    assert "unknown_block_type" in rendered
    assert "future_provider_blob" not in rendered
    assert "never-render-this-secret" not in rendered
    assert "never-render-this-payload" not in rendered


def test_release_gate_rejects_signature_field_and_schema_drift() -> None:
    trajectory = safe_trajectory()
    turns = trajectory["turns"]
    assert isinstance(turns, list)
    first_turn = turns[0]
    assert isinstance(first_turn, dict)
    first_turn["thinkingSignature"] = "never-render-this-signature"
    first_turn["unreviewed_metadata"] = {"owner": "someone"}

    report = build_trajectory_release_report([trajectory])
    rendered = json.dumps(report, ensure_ascii=False)

    assert report["passed"] is False
    assert report["opaque_reasoning_block_count"] == 1
    assert {finding["category"] for finding in report["findings"]} == {
        "forbidden_field",
        "invalid_schema",
    }
    assert "never-render-this-signature" not in rendered
    assert "unreviewed_metadata" not in rendered
    assert "someone" not in rendered


def test_release_gate_rejects_nested_signature_arguments_without_echoing_names() -> None:
    trajectory = safe_trajectory()
    turns = trajectory["turns"]
    assert isinstance(turns, list)
    assistant = turns[1]
    assert isinstance(assistant, dict)
    blocks = assistant["blocks"]
    assert isinstance(blocks, list)
    tool_call = blocks[0]
    assert isinstance(tool_call, dict)
    tool_call["arguments"] = {
        "request": {"thinking_signature": "never-render-this-nested-value"}
    }

    report = build_trajectory_release_report([trajectory])
    rendered = json.dumps(report, ensure_ascii=False)

    assert report["passed"] is False
    assert report["opaque_reasoning_block_count"] == 1
    assert "thinkingsignature" in rendered
    assert "thinking_signature" not in rendered
    assert "never-render-this-nested-value" not in rendered
