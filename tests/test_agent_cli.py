from __future__ import annotations

import json
from pathlib import Path

import pytest

from about_llm.agents import LedgerState, SQLiteLedger
from about_llm.agents.cli import (
    load_loop_checkpoint,
    load_loop_fixtures,
    load_scenario,
    main,
    run_loop_fixtures,
    run_scenario,
)

pytestmark = [
    pytest.mark.contract,
    pytest.mark.security,
    pytest.mark.integration,
]

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "projects" / "safe-agent" / "scenario.example.jsonl"
LOOP_CASES = ROOT / "projects" / "safe-agent" / "loop.example.jsonl"


def test_scenario_covers_approval_cache_and_uncertain_failure(tmp_path: Path) -> None:
    ledger = SQLiteLedger(tmp_path / "agent.db")

    report = run_scenario(load_scenario(SCENARIO), ledger, max_tool_calls=5)

    assert report["passed"] is True
    assert report["simulated_offline"] is True
    assert report["handler_attempts"] == 3
    assert [row["handler_attempted"] for row in report["outcomes"]] == [
        True,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert all(
        row["fingerprint"].startswith("sha256:") for row in report["outcomes"]
    )
    assert all(
        row["execution_fingerprint"].startswith("sha256:")
        for row in report["outcomes"]
    )
    assert report["outcomes"][0]["execution_fingerprint"] != report["outcomes"][0][
        "fingerprint"
    ]
    assert report["outcomes"][-1]["unresolved_pending"] is True
    assert report["outcomes"][2]["simulated_effect_applied"] is True
    assert report["outcomes"][2]["approval"]["simulated_unsigned_fixture"] is True
    assert report["outcomes"][4]["policy"]["reason_code"] == "missing_capability"
    assert report["outcomes"][5]["policy"]["reason_code"] == "tenant_mismatch"
    assert report["outcomes"][5]["resource"]["tenant_id"] == "tenant-b"
    assert [row["status"] for row in report["outcomes"]] == [
        "completed",
        "needs_approval",
        "completed",
        "cached",
        "policy_denied",
        "policy_denied",
        "failed",
    ]
    pending = ledger.list_stale_pending(older_than_seconds=0)
    assert [item.call_id for item in pending] == ["uncertain-1"]


@pytest.mark.smoke
def test_cli_lists_and_reconciles_pending_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger_path = tmp_path / "agent.db"
    assert (
        main(
            [
                "run",
                "--scenario",
                str(SCENARIO),
                "--ledger",
                str(ledger_path),
                "--max-tool-calls",
                "5",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["pending", "--ledger", str(ledger_path)]) == 0
    pending_payload = json.loads(capsys.readouterr().out)
    assert [item["call_id"] for item in pending_payload["pending"]] == ["uncertain-1"]

    assert (
        main(
            [
                "resolve",
                "--ledger",
                str(ledger_path),
                "--call-id",
                "uncertain-1",
                "--resolution",
                "abandoned",
                "--note",
                "Offline fixture verified: no external operation exists",
            ]
        )
        == 0
    )
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["state"] == "abandoned"
    assert resolved["reconciliation_history"][0]["resolution"] == "abandoned"
    entry = SQLiteLedger(ledger_path).lookup("uncertain-1")
    assert entry is not None
    assert entry.state is LedgerState.ABANDONED


@pytest.mark.smoke
def test_cli_evaluates_recorded_trajectory_with_explicit_denominators(
    capsys: pytest.CaptureFixture[str],
) -> None:
    traces = ROOT / "projects" / "safe-agent" / "trajectory.example.jsonl"

    assert main(["evaluate", "--traces", str(traces)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["gate_passed"] is True
    assert payload["recorded_observations_only"] is True
    assert payload["task_success"] == {
        "numerator": 3,
        "denominator": 3,
        "value": 1.0,
    }
    assert payload["blocked_unsafe_proposals"]["denominator"] == 1
    assert payload["unapproved_side_effect_attempts"]["numerator"] == 0


@pytest.mark.smoke
def test_cli_runs_typed_loop_stop_conditions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = run_loop_fixtures(load_loop_fixtures(LOOP_CASES))

    assert report["passed"] is True
    assert report["provider_usage_measured"] is False
    assert [case["termination"] for case in report["cases"]] == [
        "completed",
        "repeated_action",
        "action_cycle",
        "repeated_error",
        "needs_approval",
    ]
    assert report["cases"][0]["final_answer"] == "demo:answer"
    assert report["cases"][-1]["handler_attempts"] == 0

    assert main(["loop", "--cases", str(LOOP_CASES)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["scripted_planner"] is True


@pytest.mark.smoke
def test_cli_persists_and_resumes_approval_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "loop.db"
    checkpoint = tmp_path / "approval.checkpoint.json"
    common = [
        "--cases",
        str(LOOP_CASES),
        "--case-id",
        "approval-pause",
        "--ledger",
        str(ledger),
        "--checkpoint",
        str(checkpoint),
    ]

    assert main(["pause-loop", *common]) == 0
    paused = json.loads(capsys.readouterr().out)
    assert paused["termination"] == "needs_approval"
    assert paused["checkpoint_written_without_overwrite"] is True
    assert checkpoint.is_file()

    assert main(["resume-loop", *common]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["termination"] == "completed"
    assert resumed["model_tokens_used"] == 4
    assert resumed["cost_units_used"] == 0.2
    assert resumed["handler_attempts"] == 1
    assert resumed["simulated_unsigned_approval"] is True
    assert resumed["pause_downtime_counted_in_wall_time"] is False


@pytest.mark.parametrize(
    "record",
    [
        '{"call_id":"x","tool_name":"demo_lookup","arguments":{"key":NaN}}',
        '{"call_id":"x","tool_name":"demo_lookup","arguments":{"key":"x"},"typo":true}',
        '{"call_id":"x","call_id":"changed","tool_name":"demo_lookup","arguments":{"key":"x"}}',
    ],
)
def test_scenario_loader_rejects_nonstandard_json_and_unknown_fields(
    tmp_path: Path, record: str
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(record + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_scenario(path)


def test_loop_loader_rejects_unknown_nested_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid-loop.jsonl"
    path.write_text(
        '{"case_id":"x","context":{"task_id":"t","subject_id":"s",'
        '"tenant_id":"tenant-a","capabilities":[]},"budget":{"max_steps":1,'
        '"max_model_tokens":1,"max_cost_units":1,"max_wall_time_seconds":1,'
        '"repeated_action_limit":2,"repeated_error_limit":2},"verifier":'
        '{"expected_answer":"x","required_evidence_ids":[]},'
        '"expected_termination":"escalated","decisions":[{"decision_id":"d",'
        '"model_revision":"fixture","input_tokens":0,"output_tokens":0,'
        '"cost_units":0,"action":{"kind":"escalate","reason_code":"human",'
        '"message":"help","typo":true}}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown field"):
        load_loop_fixtures(path)


def test_checkpoint_file_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-checkpoint.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="invalid checkpoint JSON"):
        load_loop_checkpoint(path)
