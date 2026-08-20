from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [
    pytest.mark.contract,
    pytest.mark.security,
    pytest.mark.smoke,
    pytest.mark.integration,
]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "safe-agent" / "refund_lifecycle.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("agent_refund_lifecycle", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_refund_lifecycle_fences_replay_then_recovers_from_external_fact(
    tmp_path: Path,
) -> None:
    payload = _load_script().build_walkthrough(tmp_path / "refund.db")
    stages = payload["stages"]

    assert stages["schema"]["closed_schema_negative_control"] == {
        "rejected": True,
        "code": "schema_violation",
        "keyword": "additionalProperties",
    }
    assert stages["acl"]["authorized_proposal"]["status"] == "needs_approval"
    denied = stages["acl"]["cross_tenant_negative_control"]
    assert denied["status"] == "policy_denied"
    assert denied["policy_reason"] == "tenant_mismatch"

    assert stages["execution"]["status"] == "failed"
    assert stages["execution"]["local_ledger_state"] == "pending"
    assert stages["idempotency"]["handler_attempted_on_replay"] is False
    assert stages["verifier"]["status"] == "passed"

    recovery = stages["recovery"]
    assert recovery["resolution"] == "externally_confirmed"
    assert recovery["replay_after_reconciliation"]["status"] == "cached"
    assert recovery["provider_request_attempts"] == 1
    assert recovery["provider_effect_count"] == 1


def test_refund_lifecycle_cli_states_its_evidence_boundary() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout.decode("utf-8"))

    assert payload["scope"]["real_llm_or_provider_network_called"] is False
    assert payload["scope"]["planner_output_is_authored_fixture"] is True
    assert payload["scope"]["exactly_once_or_production_safety_proved"] is False
