"""Two-process CPU/Gloo control for DDP, GradScaler, and overflow consensus.

The three authored paths distinguish failures created before DDP gradient
reduction from a rank-local fault injected after reduction but before
``GradScaler.unscale_``.  The latter is deliberately not presented as normal
DDP behavior: it models application code, a custom communication path, or a
fault that can make optimizer decisions differ across ranks.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel

WORLD_SIZE = 2
INITIAL_SCALE = 8.0
BACKOFF_FACTOR = 0.5
GROWTH_INTERVAL = 1000
INITIAL_LEARNING_RATE = 0.01
CaseName = Literal[
    "pre_reduction_rank_local_overflow",
    "post_reduction_rank0_fault_without_consensus",
    "post_reduction_rank0_fault_with_global_gate",
]
CASE_NAMES: tuple[CaseName, ...] = (
    "pre_reduction_rank_local_overflow",
    "post_reduction_rank0_fault_without_consensus",
    "post_reduction_rank0_fault_with_global_gate",
)


class ScalarLinear(nn.Module):
    """A deterministic one-parameter model with an autocast-visible matmul."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.nn.functional.linear(inputs, self.weight)


def _new_scaler() -> torch.amp.GradScaler:
    scaler = torch.amp.GradScaler(
        "cpu",
        init_scale=INITIAL_SCALE,
        growth_factor=2.0,
        backoff_factor=BACKOFF_FACTOR,
        growth_interval=GROWTH_INTERVAL,
    )
    if not scaler.is_enabled():
        raise RuntimeError("CPU GradScaler is unavailable or disabled")
    return scaler


def _tensor_observation(value: Tensor) -> dict[str, bool | float | None]:
    finite = bool(torch.isfinite(value).all().item())
    return {
        "finite": finite,
        "value": float(value.item()) if finite else None,
    }


def _gradient_observation(model: ScalarLinear) -> dict[str, bool | float | None]:
    if model.weight.grad is None:
        raise AssertionError("the authored backward must populate weight.grad")
    return _tensor_observation(model.weight.grad.detach())


def _optimizer_state(
    model: ScalarLinear,
    optimizer: torch.optim.AdamW,
) -> dict[str, int | float]:
    state = optimizer.state.get(model.weight)
    if not state:
        return {"step": 0, "exp_avg": 0.0, "exp_avg_sq": 0.0}
    if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
        raise AssertionError(f"unexpected AdamW state: {sorted(state)}")
    step = state["step"]
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    if not isinstance(step, Tensor) or step.numel() != 1:
        raise AssertionError("AdamW step must be a scalar tensor")
    if not isinstance(exp_avg, Tensor) or exp_avg.shape != model.weight.shape:
        raise AssertionError("AdamW exp_avg shape drifted")
    if not isinstance(exp_avg_sq, Tensor) or exp_avg_sq.shape != model.weight.shape:
        raise AssertionError("AdamW exp_avg_sq shape drifted")
    values = (float(exp_avg.item()), float(exp_avg_sq.item()))
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("AdamW moments must remain finite in this fixture")
    return {
        "step": int(step.item()),
        "exp_avg": values[0],
        "exp_avg_sq": values[1],
    }


def _training_state(
    model: ScalarLinear,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.StepLR,
    scaler: torch.amp.GradScaler,
) -> dict[str, object]:
    parameter = float(model.weight.item())
    learning_rate = float(optimizer.param_groups[0]["lr"])
    scaler_state = scaler.state_dict()
    expected_scaler_fields = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if set(scaler_state) != expected_scaler_fields:
        raise AssertionError(
            f"unexpected GradScaler state: {sorted(scaler_state)}"
        )
    scale = float(scaler_state["scale"])
    if not all(math.isfinite(value) for value in (parameter, learning_rate, scale)):
        raise AssertionError("serialized training state must be finite")
    return {
        "parameter": parameter,
        "optimizer": _optimizer_state(model, optimizer),
        "scheduler": {
            "last_epoch": scheduler.last_epoch,
            "step_count": int(scheduler.state_dict()["_step_count"]),
        },
        "learning_rate": learning_rate,
        "grad_scaler": {
            "scale": scale,
            "growth_factor": float(scaler_state["growth_factor"]),
            "backoff_factor": float(scaler_state["backoff_factor"]),
            "growth_interval": int(scaler_state["growth_interval"]),
            "growth_tracker": int(scaler_state["_growth_tracker"]),
        },
    }


def _scaled_backward(
    ddp_model: DistributedDataParallel,
    scaler: torch.amp.GradScaler,
    *,
    loss_multiplier: float,
) -> str:
    with torch.amp.autocast(device_type="cpu", dtype=torch.float16):
        output = ddp_model(torch.ones((1, 1), dtype=torch.float32))
        loss = output.float().sum() * loss_multiplier
    scaler.scale(loss).backward()
    return str(output.dtype)


def _finite_warmup(
    ddp_model: DistributedDataParallel,
    model: ScalarLinear,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.StepLR,
    scaler: torch.amp.GradScaler,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    _scaled_backward(ddp_model, scaler, loss_multiplier=1.0)
    scaler.unscale_(optimizer)
    before_step = int(_optimizer_state(model, optimizer)["step"])
    scaler.step(optimizer)
    after_step = int(_optimizer_state(model, optimizer)["step"])
    if after_step != before_step + 1:
        raise AssertionError("finite warm-up must execute one AdamW step")
    scaler.update()
    scheduler.step()
    if float(scaler.get_scale()) != INITIAL_SCALE:
        raise AssertionError("the finite warm-up must retain the initial scale")


def _run_case(rank: int, case: CaseName) -> dict[str, object]:
    model = ScalarLinear()
    ddp_model = DistributedDataParallel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INITIAL_LEARNING_RATE,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )
    scaler = _new_scaler()

    _finite_warmup(ddp_model, model, optimizer, scheduler, scaler)
    dist.barrier()
    optimizer.zero_grad(set_to_none=True)
    state_before = _training_state(model, optimizer, scheduler, scaler)

    local_after_no_sync: dict[str, bool | float | None] | None = None
    if case == "pre_reduction_rank_local_overflow":
        with ddp_model.no_sync():
            output_dtype = _scaled_backward(
                ddp_model,
                scaler,
                loss_multiplier=float("inf") if rank == 0 else 1.0,
            )
        local_after_no_sync = _gradient_observation(model)
        output_dtype = _scaled_backward(ddp_model, scaler, loss_multiplier=1.0)
        fault = "rank0_nonfinite_loss_before_final_ddp_reduction"
    else:
        output_dtype = _scaled_backward(ddp_model, scaler, loss_multiplier=1.0)
        fault = "rank0_gradient_fill_inf_after_ddp_reduction_before_unscale"

    after_reduction = _gradient_observation(model)
    if case != "pre_reduction_rank_local_overflow" and rank == 0:
        if model.weight.grad is None:
            raise AssertionError("fault injection requires a populated gradient")
        model.weight.grad.fill_(float("inf"))
    after_fault = _gradient_observation(model)

    scale_before = float(scaler.get_scale())
    optimizer_before = _optimizer_state(model, optimizer)
    scaler.unscale_(optimizer)
    unscaled = _gradient_observation(model)
    local_nonfinite = not bool(unscaled["finite"])

    global_nonfinite: bool | None = None
    scaler_step_called = True
    if case == "post_reduction_rank0_fault_with_global_gate":
        nonfinite_flag = torch.tensor([int(local_nonfinite)], dtype=torch.int32)
        dist.all_reduce(nonfinite_flag, op=dist.ReduceOp.MAX)
        global_nonfinite = bool(nonfinite_flag.item())
        if global_nonfinite:
            scaler_step_called = False
            # GradScaler has no public API for importing another rank's
            # found-inf flag.  The application-owned gate therefore skips
            # scaler.step everywhere and applies one explicit common backoff.
            scaler.update(new_scale=scale_before * BACKOFF_FACTOR)
        else:
            scaler.step(optimizer)
            scaler.update()
    else:
        scaler.step(optimizer)
        scaler.update()

    optimizer_after_scaler = _optimizer_state(model, optimizer)
    optimizer_step_executed = int(optimizer_after_scaler["step"]) == int(
        optimizer_before["step"]
    ) + 1
    scheduler_step_called = optimizer_step_executed
    if scheduler_step_called:
        scheduler.step()

    state_after = _training_state(model, optimizer, scheduler, scaler)
    return {
        "rank": rank,
        "case": case,
        "authored_fault": fault,
        "autocast_output_dtype": output_dtype,
        "rank_local_scaled_gradient_after_no_sync": local_after_no_sync,
        "scaled_gradient_after_ddp_reduction": after_reduction,
        "scaled_gradient_after_fault_before_unscale": after_fault,
        "gradient_after_unscale_before_step": unscaled,
        "local_nonfinite_before_consensus": local_nonfinite,
        "global_nonfinite_after_max_all_reduce": global_nonfinite,
        "scaler_step_called": scaler_step_called,
        "optimizer_step_executed": optimizer_step_executed,
        "scheduler_step_called": scheduler_step_called,
        "training_state_before": state_before,
        "training_state_after": state_after,
    }


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_directory: str,
) -> None:
    if world_size != WORLD_SIZE:
        raise ValueError(f"world_size must be {WORLD_SIZE}")
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        scenarios: dict[str, object] = {}
        for case in CASE_NAMES:
            scenarios[case] = _run_case(rank, case)
            dist.barrier()
        report = {"rank": rank, "scenarios": scenarios}
        path = Path(output_directory) / f"rank-{rank}.json"
        path.write_text(
            json.dumps(report, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _scenario_reports(
    rank_reports: list[dict[str, object]],
    case: CaseName,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for rank_report in rank_reports:
        scenarios = rank_report.get("scenarios")
        if not isinstance(scenarios, dict):
            raise AssertionError("rank scenarios must be an object")
        scenario = scenarios.get(case)
        if not isinstance(scenario, dict):
            raise AssertionError(f"missing scenario {case}")
        result.append(scenario)
    return result


def _observation_finite(report: dict[str, object], field: str) -> bool:
    observation = report.get(field)
    if not isinstance(observation, dict) or not isinstance(
        observation.get("finite"), bool
    ):
        raise AssertionError(f"{field} must be a tensor observation")
    return bool(observation["finite"])


def _state_component(report: dict[str, object], when: str, field: str) -> object:
    state = report.get(when)
    if not isinstance(state, dict) or field not in state:
        raise AssertionError(f"{when}.{field} is missing")
    return state[field]


def run_control() -> dict[str, object]:
    """Execute the three real two-process controls and validate invariants."""

    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("torch.distributed with the Gloo backend is required")
    with tempfile.TemporaryDirectory(prefix="about-llm-ddp-amp-consensus-") as temp:
        root = Path(temp)
        output_directory = root / "rank-results"
        output_directory.mkdir()
        init_method = (root / "rendezvous").resolve().as_uri()
        spawn(  # type: ignore[no-untyped-call]
            _worker,
            args=(WORLD_SIZE, init_method, str(output_directory)),
            nprocs=WORLD_SIZE,
            join=True,
        )
        rank_reports: list[dict[str, object]] = [
            json.loads(
                (output_directory / f"rank-{rank}.json").read_text(encoding="utf-8")
            )
            for rank in range(WORLD_SIZE)
        ]

    pre = _scenario_reports(rank_reports, "pre_reduction_rank_local_overflow")
    local = _scenario_reports(
        rank_reports,
        "post_reduction_rank0_fault_without_consensus",
    )
    gated = _scenario_reports(
        rank_reports,
        "post_reduction_rank0_fault_with_global_gate",
    )

    all_before_states = [
        report["training_state_before"]
        for case in CASE_NAMES
        for report in _scenario_reports(rank_reports, case)
    ]
    assertions = {
        "all_cases_start_from_identical_finite_warmup_state": all(
            state == all_before_states[0] for state in all_before_states[1:]
        ),
        "pre_reduction_overflow_is_rank_local_inside_no_sync": [
            _observation_finite(report, "rank_local_scaled_gradient_after_no_sync")
            for report in pre
        ]
        == [False, True],
        "final_builtin_ddp_reduction_propagates_pre_reduction_nonfinite": [
            _observation_finite(report, "scaled_gradient_after_ddp_reduction")
            for report in pre
        ]
        == [False, False],
        "pre_reduction_overflow_skips_both_rank_optimizer_steps": [
            report["optimizer_step_executed"] for report in pre
        ]
        == [False, False],
        "pre_reduction_overflow_backs_off_both_scales": [
            _state_component(report, "training_state_after", "grad_scaler")
            for report in pre
        ]
        == [
            {
                "scale": 4.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 1000,
                "growth_tracker": 0,
            }
        ]
        * 2,
        "pre_reduction_overflow_keeps_rank_training_states_identical": (
            pre[0]["training_state_after"] == pre[1]["training_state_after"]
        ),
        "post_reduction_fault_starts_from_finite_synchronized_gradients": all(
            _observation_finite(report, "scaled_gradient_after_ddp_reduction")
            for report in (*local, *gated)
        ),
        "post_reduction_fault_is_detected_only_on_rank0_before_consensus": [
            report["local_nonfinite_before_consensus"] for report in local
        ]
        == [True, False]
        and [report["local_nonfinite_before_consensus"] for report in gated]
        == [True, False],
        "rank_local_scalers_make_different_optimizer_decisions": [
            report["optimizer_step_executed"] for report in local
        ]
        == [False, True],
        "rank_local_scalers_diverge_scale_scheduler_optimizer_and_parameter": all(
            _state_component(local[0], "training_state_after", field)
            != _state_component(local[1], "training_state_after", field)
            for field in (
                "grad_scaler",
                "scheduler",
                "optimizer",
                "parameter",
            )
        ),
        "global_max_gate_shares_nonfinite_decision": [
            report["global_nonfinite_after_max_all_reduce"] for report in gated
        ]
        == [True, True],
        "global_gate_calls_no_scaler_or_optimizer_step": [
            (report["scaler_step_called"], report["optimizer_step_executed"])
            for report in gated
        ]
        == [(False, False), (False, False)],
        "global_gate_applies_common_explicit_backoff": [
            _state_component(report, "training_state_after", "grad_scaler")
            for report in gated
        ]
        == [
            {
                "scale": 4.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 1000,
                # update(new_scale=...) does not emulate the native
                # found-inf transition that resets the growth tracker.
                "growth_tracker": 1,
            }
        ]
        * 2,
        "global_gate_keeps_rank_training_states_identical": (
            gated[0]["training_state_after"] == gated[1]["training_state_after"]
        ),
        "global_gate_keeps_model_optimizer_scheduler_equal_to_pre_step_state": all(
            _state_component(report, "training_state_before", field)
            == _state_component(report, "training_state_after", field)
            for report in gated
            for field in ("parameter", "optimizer", "scheduler", "learning_rate")
        ),
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"DDP AMP consensus control failed: {failed}")

    return {
        "implementation": "about-llm.ddp-amp-overflow-consensus-control.v1",
        "runtime": {
            "torch_version": torch.__version__,
            "backend": "gloo",
            "device": "cpu",
            "parameter_dtype": "torch.float32",
            "autocast_dtype": "torch.float16",
            "world_size": WORLD_SIZE,
            "process_start_method": "spawn",
            "rendezvous": "temporary-file-store",
            "optimizer": "torch.optim.AdamW",
            "scheduler": "torch.optim.lr_scheduler.StepLR",
            "initial_learning_rate": INITIAL_LEARNING_RATE,
            "initial_scale": INITIAL_SCALE,
            "backoff_factor": BACKOFF_FACTOR,
            "growth_interval": GROWTH_INTERVAL,
        },
        "rank_reports": rank_reports,
        "assertions": assertions,
        "scope": {
            "real_two_process_same_host_gloo_process_group_executed": True,
            "builtin_default_ddp_reducer_executed": True,
            "one_no_sync_microbatch_then_one_sync_microbatch_executed": True,
            "cpu_float16_autocast_and_cpu_grad_scaler_executed": True,
            "finite_adamw_and_steplr_warmup_executed": True,
            "pre_reduction_rank_local_nonfinite_fault_executed": True,
            "post_reduction_pre_unscale_rank0_fault_injection_executed": True,
            "rank_local_scaler_divergence_negative_control_executed": True,
            "global_nonfinite_max_all_reduce_gate_executed": True,
            "common_manual_scaler_backoff_after_global_skip_executed": True,
            "native_grad_scaler_found_inf_state_synchronized": False,
            "builtin_reducer_collective_count_directly_instrumented": False,
            "post_unscale_gradient_corruption_executed": False,
            "natural_model_or_data_induced_overflow_executed": False,
            "multiple_parameters_or_multiple_gradient_buckets_executed": False,
            "custom_ddp_comm_hook_or_conditional_parameter_graph_executed": False,
            "cuda_nccl_gpu_multi_node_or_remote_host_executed": False,
            "fsdp_zero_tensor_pipeline_or_expert_parallel_executed": False,
            "target_model_tokenizer_dataset_or_trainer_executed": False,
            "checkpoint_resume_crash_recovery_or_elastic_restart_executed": False,
            "convergence_quality_throughput_memory_or_fault_rate_proved": False,
        },
    }


def main() -> None:
    payload = run_control()
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.buffer.write((encoded + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
