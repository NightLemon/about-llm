from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langgraph")
pytest.importorskip("llama_index.core")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "safe-agent" / "framework_agent_loop_control.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("framework_agent_loop_control", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return _load_script().run_control()


def test_real_framework_loops_execute_canonical_runtime(report: dict[str, Any]) -> None:
    authorized = report["cases"]["authorized"]
    replay = report["cases"]["same-id-replay"]

    assert authorized["langchain"]["verified"] == {"passed": True, "findings": []}
    assert authorized["llamaindex"]["verified"] == {"passed": True, "findings": []}
    assert [item["status"] for item in replay["langchain"]["runtime_receipts"]] == [
        "completed",
        "cached",
    ]
    assert [item["status"] for item in replay["llamaindex"]["runtime_receipts"]] == [
        "completed",
        "cached",
    ]
    assert replay["langchain"]["handler_calls"] == ["public"]
    assert replay["llamaindex"]["handler_calls"] == ["public"]


def test_model_text_cannot_override_policy_or_unknown_tool(
    report: dict[str, Any],
) -> None:
    for case_id in ("cross-tenant", "unknown-tool"):
        for framework in ("langchain", "llamaindex"):
            case = report["cases"][case_id][framework]
            assert case["final_answer"] == "fixture:public"
            assert case["verified"]["passed"] is False
            assert case["handler_calls"] == []

    unknown = report["cases"]["unknown-tool"]
    assert unknown["langchain"]["framework_tool_results"] == [
        {
            "tool_id": "unknown-call",
            "status": "error",
            "canonical_receipt": False,
        }
    ]
    assert unknown["llamaindex"]["framework_tool_results"] == [
        {
            "tool_id": "unknown-call",
            "tool_name": "fixture_missing",
            "is_error": True,
            "canonical_call_id": None,
        }
    ]


def test_framework_call_identity_semantics_remain_explicit(
    report: dict[str, Any],
) -> None:
    replay = report["cases"]["same-id-replay"]
    assert replay["langchain"]["canonical_id_strategy"] == (
        "injected_langchain_tool_call_id"
    )
    assert all(
        receipt["framework_tool_id"] == receipt["canonical_call_id"]
        for receipt in replay["langchain"]["runtime_receipts"]
    )

    assert replay["llamaindex"]["canonical_id_strategy"] == (
        "trusted_case_and_action_hash"
    )
    assert replay["llamaindex"]["framework_tool_id_injected_into_function_tool"] is False
    derived_ids = {
        receipt["canonical_call_id"]
        for receipt in replay["llamaindex"]["runtime_receipts"]
    }
    assert len(derived_ids) == 1
    assert next(iter(derived_ids)).startswith("llamaindex-derived:")
    assert replay["llamaindex"]["dependency_warnings"][
        "pydantic_deprecation_count"
    ] > 0
    assert replay["llamaindex"]["dependency_warnings"][
        "unexpected_warning_types"
    ] == []
    assert all(report["assertions"].values())


def test_framework_agent_loop_cli_emits_closed_scope() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(completed.stdout)

    assert report["schema_version"] == "about-llm.framework-agent-loop-control.v1"
    assert report["framework_versions"] == {
        "jsonschema": version("jsonschema"),
        "langchain": version("langchain"),
        "langchain_core": version("langchain-core"),
        "langgraph": version("langgraph"),
        "llama_index_core": version("llama-index-core"),
        "pydantic": version("pydantic"),
    }
    scope = report["scope"]
    assert scope["real_langchain_create_agent_and_langgraph_loop_executed"] is True
    assert scope["real_llamaindex_function_agent_workflow_executed"] is True
    assert scope["provider_or_target_model_executed"] is False
    assert scope["framework_default_authorization_or_production_safety_proved"] is False
