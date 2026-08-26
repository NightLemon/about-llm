from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.formula, pytest.mark.contract, pytest.mark.smoke]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "inference-serving" / "generation_work_ledger.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generation_work_ledger", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qwen3_nano_vllm_work_ledger_binds_formula_to_manifest() -> None:
    payload = _load().build_ledger()

    assert payload["ledger_version"] == "about-llm.generation-work-ledger.v1"
    assert payload["manifest"]["study_id"] == "qwen3-0.6b-through-nano-vllm"
    assert payload["manifest"]["prompt_tokens"] == 768
    assert payload["manifest"]["output_tokens"] == 8
    assert payload["manifest"]["block_size_tokens"] == 256

    rows = {row["name"]: row for row in payload["scenarios"]}
    assert rows["no_prefix_reuse"] == {
        "name": "no_prefix_reuse",
        "logical_prompt_tokens": 768,
        "cached_prompt_tokens": 0,
        "scheduled_prefill_tokens": 768,
        "decode_positions": 7,
        "evaluated_forward_positions": 775,
        "positions_saved_vs_no_reuse": 0,
    }
    assert rows["exact_prefix"]["cached_prompt_tokens"] == 512
    assert rows["exact_prefix"]["scheduled_prefill_tokens"] == 256
    assert rows["exact_prefix"]["evaluated_forward_positions"] == 263
    assert rows["exact_prefix"]["positions_saved_vs_no_reuse"] == 512
    assert rows["one_token_drift"]["cached_prompt_tokens"] == 256
    assert rows["one_token_drift"]["scheduled_prefill_tokens"] == 512
    assert rows["one_token_drift"]["evaluated_forward_positions"] == 519
    assert rows["one_token_drift"]["positions_saved_vs_no_reuse"] == 256

    assert payload["scope"]["gpu_model_or_nano_vllm_engine_executed"] is False
