"""离线核对 speculative decoding 的接受概率、校正分布和 block 流程。

draft 与 target 都是直接给定的四类概率。实验先解析验证一步采样保持 target 分布，再做
Monte Carlo 直觉检查，最后构造“两 token 草稿在第二步拒绝”的 block 例子。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

import numpy as np

from about_llm.inference import (
    audit_speculative_distribution,
    speculative_sample_step,
    verify_speculative_block,
)

# draft 偏好前部 token，target 偏好后部 token，便于观察拒绝后的校正采样。
DRAFT = (0.4, 0.3, 0.2, 0.1)
TARGET = (0.1, 0.2, 0.3, 0.4)


def run_experiment(*, seed: int, trials: int) -> dict[str, Any]:
    """执行解析分布审计、随机模拟和一次强制 block 拒绝。"""

    # audit 用公式精确求和，才是分布正确性的主要证据。
    audit = audit_speculative_distribution(DRAFT, TARGET)
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(TARGET), dtype=np.int64)
    accepted = 0
    # 每次需要三个独立均匀数：draft 抽样、接受判断、拒绝后的 correction 抽样。
    for uniforms in rng.random((trials, 3)):
        result = speculative_sample_step(
            DRAFT,
            TARGET,
            draft_uniform=float(uniforms[0]),
            acceptance_uniform=float(uniforms[1]),
            correction_uniform=float(uniforms[2]),
        )
        counts[result.output_token] += 1
        accepted += int(result.accepted)
    empirical = counts / trials

    # 第二个草稿 token 在 target 下概率变低，acceptance_uniform=0.5 会触发拒绝并终止验证。
    block = verify_speculative_block(
        draft_tokens=(0, 0),
        draft_probabilities=((0.5, 0.5), (0.8, 0.2)),
        target_probabilities=((0.5, 0.5), (0.2, 0.8)),
        acceptance_uniforms=(0.0, 0.5),
        correction_uniforms=(0.0, 0.0),
        bonus_target_probabilities=(0.1, 0.9),
        bonus_uniform=0.0,
    )
    return {
        "schema_version": 1,
        "configuration": {"seed": seed, "trials": trials},
        "analytic_one_step": asdict(audit),
        "monte_carlo": {
            "empirical_output_probabilities": empirical.tolist(),
            "empirical_acceptance_rate": accepted / trials,
            "maximum_target_difference": float(
                np.max(np.abs(empirical - np.asarray(TARGET)))
            ),
        },
        "forced_block_rejection": asdict(block),
        "scope": {
            "authored_probability_vectors": True,
            "analytic_identity_checked": True,
            "monte_carlo_is_demonstration_not_proof": True,
            "model_forward_or_tokenizer_executed": False,
            "gpu_verification_kernel_executed": False,
            "latency_or_speedup_proved": False,
        },
    }


def positive_integer(value: str) -> int:
    """把 trial 数限制为正整数。"""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """读取随机种子和 Monte Carlo 次数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--trials", type=positive_integer, default=20_000)
    return parser.parse_args()


def main() -> None:
    """运行 speculative sampling 审计并输出解析值与经验频率。"""

    args = parse_args()
    print(
        json.dumps(
            run_experiment(seed=args.seed, trials=args.trials),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
