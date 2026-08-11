"""Audit exact one-step speculative sampling and block control flow offline."""

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

DRAFT = (0.4, 0.3, 0.2, 0.1)
TARGET = (0.1, 0.2, 0.3, 0.4)


def run_experiment(*, seed: int, trials: int) -> dict[str, Any]:
    audit = audit_speculative_distribution(DRAFT, TARGET)
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(TARGET), dtype=np.int64)
    accepted = 0
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
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--trials", type=positive_integer, default=20_000)
    return parser.parse_args()


def main() -> None:
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
