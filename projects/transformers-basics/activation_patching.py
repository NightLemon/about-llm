"""Deterministic MiniGPT residual-stream activation-patching experiment."""

from __future__ import annotations

import json

import torch

from about_llm.from_scratch.activation_patching import run_residual_patch_experiment
from about_llm.from_scratch.gpt_torch import GPTConfig, MiniGPT


def main() -> None:
    torch.manual_seed(23)
    model = MiniGPT(
        GPTConfig(
            vocab_size=32,
            context_length=8,
            model_dim=16,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2,
        )
    ).eval()
    clean = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    corrupted = torch.tensor([[5, 2, 3, 4]], dtype=torch.long)
    configurations = {
        "changed_source_position": (0,),
        "metric_position": (1,),
        "joint_causal_prefix": (0, 1),
        "future_position_negative_control": (2,),
    }
    results = {
        name: run_residual_patch_experiment(
            model,
            clean,
            corrupted,
            layer_index=0,
            positions=positions,
            metric_position=1,
            positive_token_id=27,
            negative_token_id=19,
        ).to_dict()
        for name, positions in configurations.items()
    }
    print(
        json.dumps(
            {
                "experiment": "seeded-random-minigpt-residual-patching-v1",
                "seed": 23,
                "clean_input_ids": clean.tolist(),
                "corrupted_input_ids": corrupted.tolist(),
                "metric_selection": (
                    "authored post-hoc token pair maximizing this fixture's "
                    "clean-minus-corrupt contrast"
                ),
                "results": results,
                "scope": {
                    "actual_forward_hooks_executed": True,
                    "clean_corrupt_batch_alignment_required": True,
                    "normalized_recovery_clipped": False,
                    "trained_language_behavior": False,
                    "target_checkpoint_tested": False,
                    "unique_natural_circuit_proved": False,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
