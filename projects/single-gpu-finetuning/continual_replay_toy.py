"""用两个会互相干扰的任务观察持续学习中的遗忘与经验回放。

实验先训练任务 A，再训练规则相反的任务 B，并比较不回放、有限 reservoir buffer
和完整回放三种策略。它会实际更新一个微型分类器，并用多随机种子与配对 bootstrap
描述结果波动；它只解释 replay 的机制和计算代价，不能证明真实 LLM 一定不会遗忘。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from about_llm.continual_learning import (
    ContinualLearningReport,
    reservoir_sample_indices,
    summarize_accuracy_matrix,
)
from about_llm.evaluation import PairedBootstrapResult, paired_bootstrap


@dataclass(frozen=True)
class ToyConfig:
    seed: int = 0
    examples_per_task: int = 256
    hidden_dim: int = 16
    task_a_steps: int = 100
    task_b_steps: int = 100
    learning_rate: float = 0.1

    def __post_init__(self) -> None:
        integers = {
            "seed": self.seed,
            "examples_per_task": self.examples_per_task,
            "hidden_dim": self.hidden_dim,
            "task_a_steps": self.task_a_steps,
            "task_b_steps": self.task_b_steps,
        }
        for name, value in integers.items():
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")


class TaskIncrementalClassifier(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.network(inputs))


def _task_data(task_id: int, examples: int) -> tuple[Tensor, Tensor]:
    """构造共享输入空间、但标签规则相反的两个任务。"""

    signal = torch.linspace(-2.0, 2.0, examples).unsqueeze(1)
    task_feature = torch.full((examples, 1), -1.0 if task_id == 0 else 1.0)
    labels = (signal[:, 0] > 0 if task_id == 0 else signal[:, 0] < 0).long()
    return torch.cat((signal, task_feature), dim=1), labels


def _fit(
    model: TaskIncrementalClassifier,
    data: tuple[Tensor, Tensor],
    *,
    steps: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    inputs, labels = data
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), labels)
        torch.autograd.backward(loss)
        optimizer.step()


@torch.inference_mode()
def _accuracy(
    model: TaskIncrementalClassifier, data: tuple[Tensor, Tensor]
) -> float:
    model.eval()
    inputs, labels = data
    return float((model(inputs).argmax(dim=1) == labels).float().mean().item())


def _prepare_after_task_a(
    config: ToyConfig,
) -> tuple[
    tuple[Tensor, Tensor],
    tuple[Tensor, Tensor],
    TaskIncrementalClassifier,
    tuple[float, float],
    tuple[float, float],
]:
    """建立相同起点并完成任务 A，供后续策略公平复用。"""

    torch.manual_seed(config.seed)
    task_a = _task_data(0, config.examples_per_task)
    task_b = _task_data(1, config.examples_per_task)
    base = TaskIncrementalClassifier(config.hidden_dim)
    baseline = (_accuracy(base, task_a), _accuracy(base, task_b))
    _fit(
        base,
        task_a,
        steps=config.task_a_steps,
        learning_rate=config.learning_rate,
    )
    after_a = (_accuracy(base, task_a), _accuracy(base, task_b))
    return task_a, task_b, base, baseline, after_a


def _train_task_b_strategy(
    base: TaskIncrementalClassifier,
    task_a: tuple[Tensor, Tensor],
    task_b: tuple[Tensor, Tensor],
    *,
    config: ToyConfig,
    pretraining_baseline: tuple[float, float],
    after_task_a: tuple[float, float],
    replay_capacity: int,
    buffer_seed: int,
) -> tuple[ContinualLearningReport, tuple[int, ...]]:
    """从同一任务 A 模型出发，用指定回放容量继续训练任务 B。"""

    # Reservoir sampling 模拟容量有限的历史样本池；容量为零时就是直接微调任务 B。
    selected = reservoir_sample_indices(
        config.examples_per_task,
        replay_capacity,
        seed=buffer_seed,
    )
    training_data = task_b
    if selected:
        # 每一步都同时看到任务 B 与选中的任务 A 样本，因此不同策略的样本计算量并不相同。
        indices = torch.tensor(selected, dtype=torch.long)
        training_data = (
            torch.cat((task_b[0], task_a[0][indices]), dim=0),
            torch.cat((task_b[1], task_a[1][indices]), dim=0),
        )
    model = copy.deepcopy(base)
    _fit(
        model,
        training_data,
        steps=config.task_b_steps,
        learning_rate=config.learning_rate,
    )
    final = (_accuracy(model, task_a), _accuracy(model, task_b))
    report = summarize_accuracy_matrix(
        (after_task_a, final),
        pretraining_baseline=pretraining_baseline,
    )
    return report, selected


def _config_payload(config: ToyConfig) -> dict[str, int | float]:
    return {
        "seed": config.seed,
        "examples_per_task": config.examples_per_task,
        "hidden_dim": config.hidden_dim,
        "task_a_steps": config.task_a_steps,
        "task_b_steps": config.task_b_steps,
        "learning_rate": config.learning_rate,
    }


def _scope_payload() -> dict[str, bool]:
    return {
        "actual_gradient_updates": True,
        "full_batch_deterministic_cpu_fixture": True,
        "replay_uses_all_old_examples": True,
        "multiple_seeds_or_confidence_intervals": False,
        "language_model_or_real_corpus": False,
        "privacy_retention_cost_modeled": False,
        "replay_always_prevents_forgetting": False,
    }


def run_experiment(config: ToyConfig | None = None) -> dict[str, Any]:
    """用一个随机种子对照完全不回放与完整回放。"""

    if config is None:
        config = ToyConfig()
    torch.use_deterministic_algorithms(True)
    task_a, task_b, base, baseline, after_a = _prepare_after_task_a(config)
    buffer_seed = config.seed + 10_000
    no_replay, _ = _train_task_b_strategy(
        base,
        task_a,
        task_b,
        config=config,
        pretraining_baseline=baseline,
        after_task_a=after_a,
        replay_capacity=0,
        buffer_seed=buffer_seed,
    )
    full_replay, _ = _train_task_b_strategy(
        base,
        task_a,
        task_b,
        config=config,
        pretraining_baseline=baseline,
        after_task_a=after_a,
        replay_capacity=config.examples_per_task,
        buffer_seed=buffer_seed,
    )
    config_payload = _config_payload(config)
    config_payload["replay_old_fraction_at_task_b"] = 0.5
    return {
        "experiment": "about-llm.task-incremental-replay-toy.v1",
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "config": config_payload,
        "task_contract": {
            "scenario": "task-incremental with an explicit task-id feature",
            "task_a": "class 1 iff signal > 0",
            "task_b": "class 1 iff signal < 0",
            "jointly_solvable_by_model_class": True,
        },
        "pretraining_baseline": list(baseline),
        "no_replay": no_replay.to_dict(),
        "full_replay": full_replay.to_dict(),
        "scope": _scope_payload(),
    }


def _strategy_payload(
    report: ContinualLearningReport,
    selected_indices: tuple[int, ...],
    *,
    config: ToyConfig,
    selection: str,
) -> dict[str, Any]:
    old_examples = len(selected_indices)
    total_examples = config.examples_per_task + old_examples
    recorded_indices: list[int] | None
    if selection == "uniform_reservoir":
        recorded_indices = list(selected_indices)
    elif selection == "none":
        recorded_indices = []
    else:
        recorded_indices = None
    return {
        "selection": selection,
        "selected_old_indices": recorded_indices,
        "selected_old_indices_representation": (
            "explicit" if recorded_indices is not None else "all indices in stream order"
        ),
        "unique_old_examples": old_examples,
        "new_examples_per_task_b_step": config.examples_per_task,
        "old_examples_per_task_b_step": old_examples,
        "total_examples_per_task_b_step": total_examples,
        "old_fraction_per_task_b_step": old_examples / total_examples,
        "task_b_optimizer_steps": config.task_b_steps,
        "new_example_presentations_at_task_b": (
            config.examples_per_task * config.task_b_steps
        ),
        "old_example_presentations_at_task_b": old_examples * config.task_b_steps,
        "metrics": report.to_dict(),
    }


def _metric_values(
    runs: Sequence[dict[str, Any]], strategy: str, metric: str
) -> list[float]:
    values: list[float] = []
    for run in runs:
        metrics = run["strategies"][strategy]["metrics"]
        if metric == "old_task_final_accuracy":
            value = metrics["accuracy_matrix"][-1][0]
        elif metric == "new_task_final_accuracy":
            value = metrics["accuracy_matrix"][-1][1]
        else:
            value = metrics[metric]
        values.append(float(value))
    return values


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _bootstrap_payload(result: PairedBootstrapResult) -> dict[str, float]:
    return {name: float(value) for name, value in asdict(result).items()}


def _paired_comparison(
    runs: Sequence[dict[str, Any]],
    strategy: str,
    *,
    confidence: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, float]]:
    comparisons: dict[str, dict[str, float]] = {}
    for metric in (
        "old_task_final_accuracy",
        "new_task_final_accuracy",
        "final_average_accuracy",
    ):
        result = paired_bootstrap(
            _metric_values(runs, "no_replay", metric),
            _metric_values(runs, strategy, metric),
            confidence=confidence,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        comparisons[f"{metric}_gain"] = _bootstrap_payload(result)
    return comparisons


def run_benchmark(
    *,
    seeds: Sequence[int] = tuple(range(20)),
    finite_buffer_capacity: int = 64,
    confidence: float = 0.95,
    bootstrap_samples: int = 5_000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """用多随机种子比较有限/完整回放，并估计相对不回放的差异。"""

    seed_tuple = tuple(seeds)
    if len(seed_tuple) < 2:
        raise ValueError("benchmark requires at least two seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seed_tuple):
        raise ValueError("benchmark seeds must be non-negative integers")
    if len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("benchmark seeds must be unique experimental units")
    template = ToyConfig()
    if (
        isinstance(finite_buffer_capacity, bool)
        or not isinstance(finite_buffer_capacity, int)
        or not 0 < finite_buffer_capacity < template.examples_per_task
    ):
        raise ValueError("finite_buffer_capacity must be between 1 and examples_per_task - 1")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be finite and in (0, 1)")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap_seed must be an integer")

    torch.use_deterministic_algorithms(True)
    runs: list[dict[str, Any]] = []
    for seed in seed_tuple:
        # 同一 seed 下三种策略共享任务 A 的模型状态，差异因此能按 seed 配对比较。
        config = ToyConfig(seed=seed)
        task_a, task_b, base, baseline, after_a = _prepare_after_task_a(config)
        buffer_seed = seed + 10_000
        strategies: dict[str, dict[str, Any]] = {}
        for name, capacity, selection in (
            ("no_replay", 0, "none"),
            ("finite_reservoir", finite_buffer_capacity, "uniform_reservoir"),
            ("full_replay", config.examples_per_task, "all_old_examples"),
        ):
            report, selected = _train_task_b_strategy(
                base,
                task_a,
                task_b,
                config=config,
                pretraining_baseline=baseline,
                after_task_a=after_a,
                replay_capacity=capacity,
                buffer_seed=buffer_seed,
            )
            strategies[name] = _strategy_payload(
                report,
                selected,
                config=config,
                selection=selection,
            )
        runs.append(
            {
                "seed": seed,
                "buffer_seed": buffer_seed,
                "pretraining_baseline": list(baseline),
                "after_task_a": list(after_a),
                "strategies": strategies,
            }
        )

    aggregate: dict[str, dict[str, float]] = {}
    # 先报告每种策略的均值，再对“候选策略 - 不回放”做成对 bootstrap。
    for strategy in ("no_replay", "finite_reservoir", "full_replay"):
        aggregate[strategy] = {
            f"mean_{metric}": _mean(_metric_values(runs, strategy, metric))
            for metric in (
                "old_task_final_accuracy",
                "new_task_final_accuracy",
                "final_average_accuracy",
                "backward_transfer",
                "average_forgetting_old_tasks",
                "forward_transfer",
            )
        }

    benchmark_config = _config_payload(template)
    del benchmark_config["seed"]
    return {
        "experiment": "about-llm.task-incremental-replay-benchmark.v1",
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "config": {
            **benchmark_config,
            "seeds": list(seed_tuple),
            "finite_buffer_capacity": finite_buffer_capacity,
            "buffer_seed_rule": "training_seed + 10000",
            "confidence": confidence,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "task_contract": {
            "scenario": "task-incremental with an explicit task-id feature",
            "task_data_fixed_across_seeds": True,
            "model_initialization_varies_by_seed": True,
            "finite_buffer_varies_by_buffer_seed": True,
            "jointly_solvable_by_model_class": True,
        },
        "runs": runs,
        "aggregate": aggregate,
        "paired_vs_no_replay": {
            strategy: _paired_comparison(
                runs,
                strategy,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            for strategy in ("finite_reservoir", "full_replay")
        },
        "bootstrap_contract": {
            "experimental_unit": "training seed paired across strategies",
            "method": "percentile bootstrap of paired seed-level metric differences",
            "metric_direction": "positive candidate-minus-no-replay gain is better",
        },
        "cost_contract": {
            "task_b_optimizer_steps_matched": True,
            "task_b_total_example_presentations_matched": False,
            "wall_time_or_energy_measured": False,
        },
        "scope": {
            "actual_gradient_updates": True,
            "finite_uniform_reservoir_executed": True,
            "full_replay_uses_all_old_examples": True,
            "old_buffer_repeated_in_every_full_batch_step": True,
            "task_or_data_distribution_resampled_across_seeds": False,
            "confidence_interval_covers_task_or_data_uncertainty": False,
            "language_model_or_real_corpus": False,
            "privacy_deletion_storage_or_energy_cost_measured": False,
            "replay_always_prevents_forgetting": False,
        },
    }


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--finite-buffer-capacity", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.benchmark:
        if args.seed_count < 2:
            raise ValueError("seed-count must be at least two")
        payload = run_benchmark(
            seeds=tuple(range(args.seed_count)),
            finite_buffer_capacity=args.finite_buffer_capacity,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    else:
        payload = run_experiment()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
