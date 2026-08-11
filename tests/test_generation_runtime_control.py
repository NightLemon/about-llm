from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "generation_runtime_control.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generation_runtime_control", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_control_executes_three_distinct_stop_paths() -> None:
    report = _load_script().run_control()
    cases = report["cases"]

    assert cases["generation_config_eos_set"]["generated_token_ids"] == [4, 3]
    assert cases["call_level_eos_override"]["generated_token_ids"] == [3, 5]
    assert cases["call_level_length_cap"]["generated_token_ids"] == [4, 6]
    assert report["assertions"] == {
        "generation_config_eos_list_stopped_on_member_3": True,
        "call_eos_5_overrode_config_eos_set_2_3": True,
        "config_eos_3_did_not_stop_override_case": True,
        "call_max_new_tokens_2_stopped_without_eos": True,
        "caller_generation_config_not_mutated": True,
    }
    assert report["scope"] == {
        "real_transformers_generation_mixin_executed": True,
        "real_tiny_gpt2_forward_executed": True,
        "authored_logits_processor_overrode_all_next_token_scores": True,
        "random_untrained_model_used": True,
        "real_tokenizer_or_chat_template_executed": False,
        "public_checkpoint_or_remote_code_loaded": False,
        "vllm_or_provider_runtime_executed": False,
        "model_quality_latency_throughput_or_gpu_behavior_proved": False,
        "provider_style_finish_reason_observed": False,
    }


def test_runtime_control_is_deterministic_and_records_processor_steps() -> None:
    first = _load_script().run_control()
    second = _load_script().run_control()

    for report in (first, second):
        report.pop("torch_version")
        report.pop("transformers_version")
    assert first == second
    cases = first["cases"]
    assert cases["generation_config_eos_set"]["processor_trace"] == [
        {"generation_step": 0, "input_length_before_step": 2, "forced_token_id": 4},
        {"generation_step": 1, "input_length_before_step": 3, "forced_token_id": 3},
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt_length": 0, "token_plan": (1,), "vocabulary_size": 4}, "prompt"),
        ({"prompt_length": 1, "token_plan": (), "vocabulary_size": 4}, "empty"),
        (
            {"prompt_length": 1, "token_plan": (4,), "vocabulary_size": 4},
            "inside the vocabulary",
        ),
    ],
)
def test_forced_plan_rejects_invalid_contract(
    kwargs: dict[str, object], message: str
) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match=message):
        module.ForcedTokenPlan(**kwargs)


def test_runtime_control_cli_emits_same_evidence() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    report = json.loads(completed.stdout)

    assert report["cases"]["call_level_eos_override"]["generated_token_ids"] == [
        3,
        5,
    ]
    assert report["scope"]["public_checkpoint_or_remote_code_loaded"] is False
