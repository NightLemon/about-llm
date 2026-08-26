from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "trace_rmsnorm_operator_stack.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("trace_rmsnorm_operator_stack", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.formula
def test_trace_connects_layout_math_fx_and_aten_without_kernel_overclaim() -> None:
    trace: dict[str, Any] = _load().build_trace()

    tensor = trace["tensor_contract"]
    assert tensor["base_shape"] == [2, 3, 4]
    assert tensor["base_stride"] == [12, 4, 1]
    assert tensor["view_shape"] == [3, 2, 4]
    assert tensor["view_stride"] == [4, 12, 1]
    assert tensor["view_is_contiguous"] is False
    assert tensor["view_shares_storage_with_base"] is True
    assert tensor["contiguous_stride"] == [8, 4, 1]
    assert tensor["contiguous_is_contiguous"] is True
    assert tensor["contiguous_shares_storage_with_view"] is False
    assert tensor["logical_bytes"] == 96

    contract = trace["rmsnorm_contract"]
    assert contract["max_abs_error_vs_torch_functional"] == pytest.approx(0.0, abs=1e-6)
    assert contract["input_gradient_finite"] is True
    assert contract["weight_gradient_finite"] is True

    fx_targets = [item["target"] for item in trace["graphs"]["fx_python_graph"]]
    assert fx_targets == [
        "operator.mul",
        "mean",
        "operator.add",
        "torch.rsqrt",
        "operator.mul",
        "operator.mul",
    ]
    export_targets = [item["target"] for item in trace["graphs"]["export_aten_graph"]]
    assert export_targets == [
        "aten.mul.Tensor",
        "aten.mean.dim",
        "aten.add.Tensor",
        "aten.rsqrt.default",
        "aten.mul.Tensor",
        "aten.mul.Tensor",
    ]

    assert trace["scope"] == {
        "mathematical_parity_checked": True,
        "non_contiguous_layout_executed": True,
        "forward_and_backward_executed": True,
        "torch_export_executed": True,
        "torch_compile_executed": False,
        "cuda_or_accelerator_executed": False,
        "kernel_count_inferred_from_fx_or_export": False,
        "cross_platform_support_or_performance_proven": False,
    }


def test_optional_profile_reports_observed_aten_events_without_treating_them_as_kernels() -> None:
    trace: dict[str, Any] = _load().build_trace(profile=True)
    profiler = trace["profiler"]

    assert profiler["executed"] is True
    reference = {event["operator"] for event in profiler["reference_decomposition"]}
    framework = {event["operator"] for event in profiler["framework_rms_norm"]}
    assert {"aten::mul", "aten::mean", "aten::rsqrt"} <= reference
    assert "aten::rms_norm" in framework
    assert "不能直接当作" in profiler["interpretation"]


def test_support_probe_validates_shape_dtype_and_cuda_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()

    with pytest.raises(ValueError, match="must be positive"):
        module.build_trace(hidden_size=0)
    with pytest.raises(ValueError, match="dtype_name"):
        module.build_trace(dtype_name="float64")
    if not module.torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            module.build_trace(device_name="cuda")

    monkeypatch.delattr(module.functional, "rms_norm")
    with pytest.raises(RuntimeError, match=r"requires PyTorch 2\.4 or later"):
        module.build_trace()


@pytest.mark.smoke
def test_guided_and_json_views_explain_the_evidence_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load()

    assert module.main([]) == 0
    guided = capsys.readouterr().out
    assert "跟着一次 RMSNorm 看懂算子计算栈" in guided
    assert "Module、Python/FX 节点和 ATen 算子不是同一抽象层" in guided
    assert "图节点仍然不是 GPU kernel" in guided

    assert module.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "about-llm.rmsnorm-operator-stack.v1"
    assert payload["scope"]["torch_compile_executed"] is False
