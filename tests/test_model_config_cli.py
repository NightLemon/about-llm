from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "inspect_config.py"
CONFIGS = ROOT / "projects" / "transformers-basics" / "configs"


def _run(config_name: str, *arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(CONFIGS / config_name), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise AssertionError("CLI must emit one JSON object")
    return payload


def test_standard_gqa_cli_emits_exact_scenario_ledger() -> None:
    payload = _run(
        "standard-gqa.example.json",
        "--tokens",
        "4096",
        "--batch-size",
        "1",
        "--element-bytes",
        "2",
    )

    assert payload["estimate_refused"] is False
    assert payload["estimate_refusal_reason"] is None
    assert payload["inspection"]["model_type"] == "authored_standard_gqa"
    assert payload["standard_kv_estimates"] == [
        {
            "batch_size": 1,
            "bytes_per_token_per_layer": 4096,
            "element_bytes": 2,
            "ideal_tensor_payload_only": True,
            "includes_allocator_metadata_alignment_workspace_or_scales": False,
            "token_count": 4096,
            "total_bytes": 536_870_912,
        }
    ]


def test_mla_cli_refuses_standard_kv_formula() -> None:
    payload = _run("mla-moe.example.json", "--tokens", "4096")

    assert payload["estimate_refused"] is True
    assert payload["standard_kv_estimates"] == []
    assert "standard dense K/V formula must not be applied" in payload[
        "estimate_refusal_reason"
    ]
    assert payload["inspection"]["known_mla_markers_present"] is True
