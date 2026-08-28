# ruff: noqa: RUF001 -- Full-width punctuation is intentional in learner output.
"""沿一次 RMSNorm 调用追踪张量布局、计算图、ATen 算子和 profiler。

同一个数学公式会在 PyTorch 中经过多个抽象层。本实验先构造非连续张量，再比较手写
RMSNorm 与框架实现，接着查看 FX 图和 export 后的 ATen 图，最后可选地记录真实运行事件。
重点是理解各层的边界，以及图节点与 GPU kernel 之间并非一一对应的关系。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

SCHEMA_VERSION = "about-llm.rmsnorm-operator-stack.v1"
DEFAULT_BATCH_SIZE = 2
DEFAULT_SEQUENCE_LENGTH = 3
DEFAULT_HIDDEN_SIZE = 4
EPSILON = 1e-6


class ReferenceRMSNorm(nn.Module):
    """故意拆成基础算子的 RMSNorm，便于阅读公式和计算图。"""

    def __init__(self, hidden_size: int) -> None:
        """创建一组非全 1 权重，使逐元素缩放在结果中可观察。"""

        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, steps=hidden_size))

    def forward(self, inputs: Tensor) -> Tensor:
        """按最后一维计算均方根并应用可学习权重。"""

        # keepdim=True 保留归一化维度，结果才能沿 hidden 维广播回输入 shape。
        mean_square = (inputs * inputs).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(mean_square + EPSILON)
        return normalized * self.weight


def _target_name(target: object) -> str:
    """把 FX 节点 target 规范化成适合报告展示的稳定名称。"""

    if isinstance(target, str):
        return target
    rendered = str(target)
    if rendered.startswith("aten."):
        return rendered
    name = getattr(target, "__name__", None)
    module = getattr(target, "__module__", None)
    if isinstance(name, str) and isinstance(module, str):
        if module == "_operator":
            module = "operator"
        return f"{module}.{name}"
    return rendered


def _graph_operations(graph_module: torch.fx.GraphModule) -> list[dict[str, str]]:
    """只保留计算相关节点，省略 placeholder 和 output 等结构节点。"""

    return [
        {"node_kind": node.op, "target": _target_name(node.target)}
        for node in graph_module.graph.nodes
        if node.op in {"call_function", "call_method", "call_module"}
    ]


def _profile_call(
    call: Callable[[], Tensor], *, device: torch.device
) -> list[dict[str, int | str]]:
    """执行一次函数并汇总当前设备实际观察到的 ATen 事件。"""

    # CUDA launch 是异步的，前后同步确保 profiler 覆盖本次调用的完整工作。
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize(device)
    with torch.profiler.profile(activities=activities) as profiler:
        call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return sorted(
        (
            {"operator": event.key, "calls": event.count}
            for event in profiler.key_averages()
            if event.key.startswith("aten::")
        ),
        key=lambda item: str(item["operator"]),
    )


def build_trace(
    *,
    profile: bool = False,
    device_name: str = "cpu",
    dtype_name: str = "float32",
    batch_size: int = DEFAULT_BATCH_SIZE,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
) -> dict[str, object]:
    """构造输入并收集 RMSNorm 各抽象层的可核对证据。"""

    # functional.rms_norm 是本实验的框架参考；旧 PyTorch 没有该接口时直接说明要求。
    framework_rms_norm = getattr(functional, "rms_norm", None)
    if framework_rms_norm is None:
        raise RuntimeError(
            "this experiment requires PyTorch 2.4 or later for torch.nn.functional.rms_norm"
        )
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device_name must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; rerun with --device cpu")
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError("dtype_name must be float32, float16, or bfloat16")
    if min(batch_size, sequence_length, hidden_size) <= 0:
        raise ValueError("batch_size, sequence_length, and hidden_size must be positive")
    # 通过白名单把命令行字符串映射为实际 device 和 dtype，避免隐式转换。
    device = torch.device(device_name)
    dtype = dtype_by_name[dtype_name]
    element_count = batch_size * sequence_length * hidden_size
    # 连续 base 提供可预测数值；transpose 只改变 view 的 shape/stride，并共享 storage。
    base = torch.linspace(-1.0, 1.0, steps=element_count, dtype=dtype, device=device).reshape(
        batch_size,
        sequence_length,
        hidden_size,
    )
    transposed = base.transpose(0, 1)
    # contiguous() 会在布局不连续时物化新 storage，作为内存布局对照。
    contiguous = transposed.contiguous()

    # 前向阶段比较同一输入、权重和 epsilon 下的手写公式与框架算子。
    module = ReferenceRMSNorm(hidden_size).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        reference_output = module(transposed)
        framework_output = framework_rms_norm(
            transposed,
            (hidden_size,),
            module.weight,
            EPSILON,
        )

    # 另建需要梯度的叶子张量，真实执行一次 backward 并检查梯度没有 NaN/Inf。
    gradient_input = transposed.detach().clone().requires_grad_(True)
    gradient_weight = module.weight.detach().clone().requires_grad_(True)
    mean_square = (gradient_input * gradient_input).mean(dim=-1, keepdim=True)
    gradient_output = gradient_input * torch.rsqrt(mean_square + EPSILON) * gradient_weight
    loss = gradient_output.square().mean()
    input_gradient, weight_gradient = torch.autograd.grad(
        loss,
        (gradient_input, gradient_weight),
    )

    # FX 保留较高层 Python 运算；torch.export 通常进一步落到 ATen 运算。
    fx_graph = torch.fx.symbolic_trace(module)
    exported_program = torch.export.export(module, (transposed,))

    profiler_report: dict[str, object]
    if profile:
        # profiler 记录的是当前 PyTorch build 与设备的动态事件，不代表所有平台。
        profiler_report = {
            "executed": True,
            "device": device.type,
            "reference_decomposition": _profile_call(lambda: module(transposed), device=device),
            "framework_rms_norm": _profile_call(
                lambda: framework_rms_norm(
                    transposed,
                    (hidden_size,),
                    module.weight,
                    EPSILON,
                ),
                device=device,
            ),
            "interpretation": (
                "这些事件来自当前安装的 PyTorch 和当前设备；"
                "它们不能直接当作其他版本或设备的 GPU kernel 清单。"
            ),
        }
    else:
        profiler_report = {
            "executed": False,
            "run_hint": "rerun with --profile to collect ATen operator events",
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "environment": {
            "torch_version": torch.__version__,
            "execution_device": device.type,
            "dtype": dtype_name,
            "torch_cuda_version": torch.version.cuda,
            "accelerator_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "tensor_contract": {
            "base_shape": list(base.shape),
            "base_stride": list(base.stride()),
            "view_shape": list(transposed.shape),
            "view_stride": list(transposed.stride()),
            "view_is_contiguous": transposed.is_contiguous(),
            "view_shares_storage_with_base": (
                transposed.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
            ),
            "contiguous_stride": list(contiguous.stride()),
            "contiguous_is_contiguous": contiguous.is_contiguous(),
            "contiguous_shares_storage_with_view": (
                contiguous.untyped_storage().data_ptr() == transposed.untyped_storage().data_ptr()
            ),
            "dtype": str(transposed.dtype).removeprefix("torch."),
            "logical_elements": transposed.numel(),
            "logical_bytes": transposed.numel() * transposed.element_size(),
        },
        "rmsnorm_contract": {
            "formula": "y = x * rsqrt(mean(x^2, dim=-1) + eps) * weight",
            "normalized_dimension": hidden_size,
            "epsilon": EPSILON,
            "output_shape": list(reference_output.shape),
            "max_abs_error_vs_torch_functional": float(
                (reference_output - framework_output).abs().max().item()
            ),
            "input_gradient_finite": bool(torch.isfinite(input_gradient).all().item()),
            "weight_gradient_finite": bool(torch.isfinite(weight_gradient).all().item()),
        },
        "graphs": {
            "fx_python_graph": _graph_operations(fx_graph),
            "export_aten_graph": _graph_operations(exported_program.graph_module),
        },
        "profiler": profiler_report,
        "scope": {
            "mathematical_parity_checked": True,
            "non_contiguous_layout_executed": True,
            "forward_and_backward_executed": True,
            "torch_export_executed": True,
            "torch_compile_executed": False,
            "cuda_or_accelerator_executed": device.type == "cuda",
            "kernel_count_inferred_from_fx_or_export": False,
            "cross_platform_support_or_performance_proven": False,
        },
    }


def _operator_names(records: object) -> str:
    """把图节点名称连接成一条便于阅读的调用链。"""

    if not isinstance(records, list):
        raise TypeError("operator records must be a list")
    names = [str(record["target"]) for record in records if isinstance(record, dict)]
    return " → ".join(names)


def _profile_lines(profiler: dict[str, object]) -> list[str]:
    """将可选 profiler 结果转换为一行中文说明。"""

    if profiler["executed"] is not True:
        return ["未运行 profiler；加入 --profile 可观察当前 PyTorch 的 ATen 事件。"]
    framework_events = profiler["framework_rms_norm"]
    if not isinstance(framework_events, list):
        raise TypeError("framework profiler events must be a list")
    rendered = ", ".join(
        f"{event['operator']}×{event['calls']}"
        for event in framework_events
        if isinstance(event, dict)
    )
    return [f"当前环境观察到的 framework RMSNorm 事件: {rendered}"]


def render_trace(trace: dict[str, object]) -> str:
    """按“布局→数学→计算图→运行事件”的顺序渲染 trace。"""

    environment = trace["environment"]
    tensor = trace["tensor_contract"]
    contract = trace["rmsnorm_contract"]
    graphs = trace["graphs"]
    profiler = trace["profiler"]
    if not all(
        isinstance(section, dict) for section in (environment, tensor, contract, graphs, profiler)
    ):
        raise TypeError("trace sections must be objects")

    environment = dict(environment)
    tensor = dict(tensor)
    contract = dict(contract)
    graphs = dict(graphs)
    profiler = dict(profiler)
    # 先讲每层观察到了什么，再提醒下一层还可能继续分解或融合。
    lines = [
        "跟着一次 RMSNorm 看懂算子计算栈",
        "",
        "1. 同一批数值，先看张量布局",
        f"base shape/stride: {tensor['base_shape']} / {tensor['base_stride']}",
        f"transpose view shape/stride: {tensor['view_shape']} / {tensor['view_stride']}",
        f"view contiguous: {tensor['view_is_contiguous']}",
        f"view 与 base 共享 storage: {tensor['view_shares_storage_with_base']}",
        f"contiguous copy stride: {tensor['contiguous_stride']}",
        "结论: transpose 可以只改 shape/stride；contiguous 会在需要时物化新布局。",
        "",
        "2. 数学语义保持不变",
        f"RMSNorm: {contract['formula']}",
        f"reference 与 torch.nn.functional.rms_norm 最大绝对误差: "
        f"{float(contract['max_abs_error_vs_torch_functional']):.3e}",
        f"输入梯度和权重梯度均为有限值: "
        f"{contract['input_gradient_finite'] and contract['weight_gradient_finite']}",
        "",
        "3. 同一个 Python 函数可以看到两层图",
        f"FX 图: {_operator_names(graphs['fx_python_graph'])}",
        f"torch.export 的 ATen 图: {_operator_names(graphs['export_aten_graph'])}",
        "结论: Module、Python/FX 节点和 ATen 算子不是同一抽象层。",
        "",
        "4. 图节点仍然不是 GPU kernel",
        *_profile_lines(profiler),
        "编译器可以继续分解或融合这些节点，后端还会根据设备、数据类型、形状"
        "和内存布局选择实现。",
        "",
        f"本次在 {environment['execution_device']} 上核对了数学结果、布局、FX/ATen 图"
        "和可选 profiler 事件。",
        "torch.compile、性能测量和完整平台支持需要按后续实验单独验证。",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """定义设备、dtype、shape、profiler 和 JSON 输出选项。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="collect CPU ATen operator events for the current PyTorch build",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="execute the tensor path on CPU or an available CUDA device",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="input and weight dtype for this support probe",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete machine-readable trace",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """收集 RMSNorm trace，并打印学习导览或完整 JSON。"""

    # 人类视图包含中文，统一终端编码后再输出。
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    trace = build_trace(
        profile=args.profile,
        device_name=args.device,
        dtype_name=args.dtype,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
    )
    if args.json:
        print(json.dumps(trace, ensure_ascii=False, allow_nan=False, indent=2))
    else:
        print(render_trace(trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
