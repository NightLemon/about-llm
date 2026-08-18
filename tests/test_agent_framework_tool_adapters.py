from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.security, pytest.mark.integration]

pytest.importorskip("langchain_core")
pytest.importorskip("llama_index.core")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "safe-agent" / "framework_tool_adapter_control.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("framework_tool_adapter_control", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_framework_tools_transport_proposals_into_canonical_runtime() -> None:
    report = _load_script().run_control()

    assert report["cases"]["authorized"]["langchain"]["status"] == "completed"
    assert report["cases"]["authorized"]["llamaindex"]["status"] == "completed"
    assert report["cases"]["authorized"]["langchain"]["value"] == {
        "value": "fixture:public"
    }
    assert report["cases"]["same_call_id_replay"]["langchain"]["status"] == "cached"
    assert report["cases"]["same_call_id_replay"]["llamaindex"]["status"] == "cached"
    assert report["cases"]["cross_tenant"]["langchain"]["status"] == "policy_denied"
    assert report["cases"]["cross_tenant"]["llamaindex"]["status"] == "policy_denied"
    assert all(report["assertions"].values())


def test_direct_framework_schema_behavior_is_not_overclaimed() -> None:
    report = _load_script().run_control()

    assert report["cases"]["invalid_type"] == {
        "langchain_rejection": "ValidationError",
        "llamaindex_canonical_rejection": "ToolArgumentValidationError",
    }
    assert report["contract"]["model_visible_fields"] == ["key"]
    assert report["contract"]["trusted_context_model_visible"] is False
    assert report["scope"]["framework_default_authorization_or_production_safety_proved"] is False
    assert report["scope"]["langgraph_or_llamaindex_agent_loop_executed"] is False

