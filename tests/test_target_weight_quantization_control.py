from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from about_llm.inference.quantized_bundle import QuantizedBundleIdentity
from about_llm.integrations.transformers_weight_quantization_control import (
    TARGET_MANIFEST_FINGERPRINT,
    execute_selected_linear_quantization,
    verify_recorded_target_weight_quantization_report,
)
from about_llm.llmops import artifact_fingerprint

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRECTORY = ROOT / "projects" / "transformers-basics" / "target-checkpoints"
RECORDED_REPORT = (
    TARGET_DIRECTORY / "qwen2.5-0.5b-instruct.weight-int4.recorded-report.json"
)
SCRIPT = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "run_qwen_weight_quantization_control.py"
)


class TinyLinearLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.block = torch.nn.Module()
        self.block.proj = torch.nn.Linear(4, 4, bias=False)
        self.head = torch.nn.Linear(4, 16, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> Any:
        assert use_cache is False
        assert return_dict is True
        hidden = torch.tanh(self.block.proj(self.embedding(input_ids)))
        return SimpleNamespace(logits=self.head(hidden))


def _identity() -> QuantizedBundleIdentity:
    return QuantizedBundleIdentity(
        model_family="fixture/tiny-linear-lm",
        model_revision="fixture-revision",
        tokenizer_id="fixture/tokenizer",
        tokenizer_revision="fixture-tokenizer-revision",
        architecture_config={"hidden_size": 4, "selected_module": "block.proj"},
    )


def test_selected_linear_control_restores_model_and_round_trips_artifact() -> None:
    torch.manual_seed(20260815)
    model = TinyLinearLM()
    model.train()
    original_weight = model.block.proj.weight.detach().clone()

    report = execute_selected_linear_quantization(
        model,
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        module_name="block.proj",
        bundle_identity=_identity(),
        bit_width=4,
        group_size=2,
    )

    assert report["weight_shape"] == [4, 4]
    assert report["groups_per_row"] == 2
    assert report["selected_parameters"] == 16
    assert report["artifact"]["round_trip_codes_equal"] is True
    assert report["artifact"]["round_trip_scales_equal"] is True
    assert report["artifact"]["tamper_rejected_before_decode"] is True
    assert report["execution"][
        "in_memory_vs_reloaded_selected_output_exact"
    ] is True
    assert report["execution"]["source_weight_restored_exactly"] is True
    assert torch.equal(model.block.proj.weight, original_weight)
    assert model.training is True
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_selected_linear_control_rejects_bias_and_missing_module() -> None:
    model = TinyLinearLM()
    input_ids = torch.tensor([[1]], dtype=torch.long)
    with pytest.raises(ValueError, match=r"existing torch\.nn\.Linear"):
        execute_selected_linear_quantization(
            model,
            input_ids=input_ids,
            module_name="missing",
            bundle_identity=_identity(),
        )
    model.block.proj = torch.nn.Linear(4, 4, bias=True)
    with pytest.raises(ValueError, match="bias-free"):
        execute_selected_linear_quantization(
            model,
            input_ids=input_ids,
            module_name="block.proj",
            bundle_identity=_identity(),
        )


def test_recorded_target_run_is_closed_and_scope_limited() -> None:
    report = verify_recorded_target_weight_quantization_report(RECORDED_REPORT)

    assert report["manifest_fingerprint"] == TARGET_MANIFEST_FINGERPRINT
    assert report["report_fingerprint"] == (
        "sha256:df9ee045be4bf2e2ab4441bacfe24ffd1f903e9a0715bda0f35219ac3928f5cb"
    )
    assert report["selection"]["selected_parameters"] == 802_816
    assert report["selection"]["total_model_parameters"] == 494_032_768
    assert report["artifact"]["serialized_bundle_bytes"] == 427_328
    assert report["artifact"]["serialized_compression_ratio"] == pytest.approx(
        7.514752134192002
    )
    assert report["execution"]["last_logits_error"][
        "relative_l2_error"
    ] == pytest.approx(0.08513807180570929)
    assert report["execution"]["last_argmax_match"] is True
    assert report["scope"]["full_checkpoint_quantized"] is False
    assert report["scope"]["fused_low_bit_kernel_executed"] is False
    assert report["scope"]["model_quality_or_effective_context_proven"] is False


def test_recorded_report_tampering_is_rejected(tmp_path: Path) -> None:
    report = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))
    report["execution"]["last_argmax_match"] = False
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="report fingerprint mismatch"):
        verify_recorded_target_weight_quantization_report(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["scope"].update(
                {"full_checkpoint_quantized": True}
            ),
            r"report.scope.full_checkpoint_quantized drift",
        ),
        (
            lambda report: report["selection"].update(
                {"selected_parameters": 494_032_768}
            ),
            r"report.selection does not match",
        ),
        (
            lambda report: report["artifact"].update(
                {"serialized_compression_ratio": 100.0}
            ),
            r"compression ratio is inconsistent",
        ),
    ],
)
def test_cooperatively_rehashed_scope_or_arithmetic_drift_is_rejected(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    report = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))
    mutation(report)
    projection = dict(report)
    del projection["report_fingerprint"]
    report["report_fingerprint"] = "sha256:" + artifact_fingerprint(projection)
    path = tmp_path / "cooperatively-rehashed.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_recorded_target_weight_quantization_report(path)


def test_cli_verifies_recorded_report_without_loading_weights() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--verify", str(RECORDED_REPORT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(completed.stdout)

    assert report["report_fingerprint"].startswith("sha256:")
    assert report["scope"]["quantized_runtime_loaded"] is False
