"""Local text-tokenizer PPO control with EOS, truncation, and padding semantics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from about_llm.finetuning.ppo_text import (
    TEXT_CONTROL_CHAT_TEMPLATE,
    TEXT_CONTROL_HORIZON,
    TEXT_CONTROL_MESSAGES,
    build_text_contexts,
    build_text_control_tokenizer,
    collect_autoregressive_text_rollout,
    render_text_control_prompt,
)
from about_llm.finetuning.ppo_transformer import (
    TinyCausalActorCritic,
    TransformerRollout,
    finite_non_negative,
    freeze_copy,
    optimize_token_rollout,
    positive_integer,
)

HORIZON = TEXT_CONTROL_HORIZON
CHAT_TEMPLATE = TEXT_CONTROL_CHAT_TEMPLATE
MESSAGES = TEXT_CONTROL_MESSAGES


@dataclass(frozen=True)
class ExactTextObjectives:
    """Exactly enumerated task metrics for the two-token generation tree."""

    expected_task_reward: float
    good_first_probability: float
    eos_after_good_probability: float
    good_then_eos_probability: float


def build_tokenizer() -> PreTrainedTokenizerFast:
    return build_text_control_tokenizer()


def render_prompt(tokenizer: PreTrainedTokenizerFast) -> tuple[str, tuple[int, ...]]:
    return render_text_control_prompt(tokenizer)


def _contexts(
    prompt_ids: tuple[int, ...],
    generated_prefixes: Tensor,
    *,
    max_context_length: int,
) -> tuple[Tensor, Tensor]:
    return build_text_contexts(
        prompt_ids,
        generated_prefixes,
        max_context_length=max_context_length,
        pad_token_id=1,
    )


@torch.inference_mode()
def exact_text_objectives(
    model: TinyCausalActorCritic,
    prompt_ids: tuple[int, ...],
    *,
    good_token_id: int,
    eos_token_id: int,
    vocab_size: int,
    max_context_length: int,
) -> ExactTextObjectives:
    """Compute exact dense reward and exact `good, EOS` success probability."""

    model.eval()
    empty_prefix = torch.empty((1, 0), dtype=torch.long)
    first_context, first_mask = _contexts(
        prompt_ids, empty_prefix, max_context_length=max_context_length
    )
    first_logits, _ = model.forward_contexts(first_context, first_mask)
    first_probabilities = torch.softmax(first_logits[0], dim=-1)
    first_actions = torch.arange(vocab_size, dtype=torch.long).unsqueeze(1)
    second_contexts, second_masks = _contexts(
        prompt_ids, first_actions, max_context_length=max_context_length
    )
    second_logits, _ = model.forward_contexts(second_contexts, second_masks)
    second_eos_probabilities = torch.softmax(second_logits, dim=-1)[:, eos_token_id]
    continuation = torch.ones(vocab_size, dtype=torch.float32)
    continuation[eos_token_id] = 0
    expected_second_reward = torch.sum(
        first_probabilities * continuation * second_eos_probabilities
    )
    expected_reward = (
        first_probabilities[good_token_id] + expected_second_reward
    )
    good_first_probability = first_probabilities[good_token_id]
    eos_after_good_probability = second_eos_probabilities[good_token_id]
    exact_success = good_first_probability * eos_after_good_probability
    return ExactTextObjectives(
        expected_task_reward=float(expected_reward),
        good_first_probability=float(good_first_probability),
        eos_after_good_probability=float(eos_after_good_probability),
        good_then_eos_probability=float(exact_success),
    )


@torch.inference_mode()
def collect_text_rollout(
    model: TinyCausalActorCritic,
    reference: TinyCausalActorCritic,
    *,
    prompt_ids: tuple[int, ...],
    good_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    episodes: int,
    kl_coefficient: float,
    generator: torch.Generator,
) -> TransformerRollout:
    def dense_good_eos_reward(actions: Tensor, valid_mask: Tensor) -> Tensor:
        rewards = torch.zeros(actions.shape, dtype=torch.float32)
        rewards[:, 0] = (
            valid_mask[:, 0] & (actions[:, 0] == good_token_id)
        ).to(torch.float32)
        rewards[:, 1] = (
            valid_mask[:, 1] & (actions[:, 1] == eos_token_id)
        ).to(torch.float32)
        return rewards

    return collect_autoregressive_text_rollout(
        model,
        reference,
        prompt_ids=prompt_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
        horizon=HORIZON,
        episodes=episodes,
        kl_coefficient=kl_coefficient,
        task_reward_fn=dense_good_eos_reward,
        generator=generator,
    )


def run_smoke(
    *,
    iterations: int = 8,
    episodes_per_iteration: int = 128,
    epochs: int = 3,
    minibatch_size: int = 64,
    learning_rate: float = 0.01,
    kl_coefficient: float = 0.01,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    bootstrap_truncated: bool = False,
) -> dict[str, Any]:
    iteration_count = positive_integer(iterations, "iterations")
    episodes = positive_integer(episodes_per_iteration, "episodes_per_iteration")
    epoch_count = positive_integer(epochs, "epochs")
    batch_size = positive_integer(minibatch_size, "minibatch_size")
    rate = finite_non_negative(learning_rate, "learning_rate")
    if rate == 0:
        raise ValueError("learning_rate must be positive")
    coefficient = finite_non_negative(kl_coefficient, "kl_coefficient")
    value_weight = finite_non_negative(value_coefficient, "value_coefficient")
    entropy_weight = finite_non_negative(entropy_coefficient, "entropy_coefficient")
    if not isinstance(bootstrap_truncated, bool):
        raise ValueError("bootstrap_truncated must be boolean")
    torch.manual_seed(71)
    rollout_generator = torch.Generator(device="cpu").manual_seed(72)
    optimizer_generator = torch.Generator(device="cpu").manual_seed(73)
    tokenizer = build_tokenizer()
    rendered_prompt, prompt_ids = render_prompt(tokenizer)
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise AssertionError("tokenizer must define EOS and padding IDs")
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    good_token_id = tokenizer.convert_tokens_to_ids("good")
    vocab_size = len(tokenizer)
    max_context_length = len(prompt_ids) + HORIZON
    model = TinyCausalActorCritic(
        vocab_size=vocab_size,
        max_positions=max_context_length,
        bos_token_id=eos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        separate_value_backbone=True,
    ).to("cpu")
    reference = freeze_copy(model)
    reference_before = {
        name: parameter.detach().clone()
        for name, parameter in reference.named_parameters()
    }
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    initial_exact = exact_text_objectives(
        model,
        prompt_ids,
        good_token_id=good_token_id,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        max_context_length=max_context_length,
    )
    reports: list[dict[str, Any]] = []
    for iteration in range(iteration_count):
        rollout = collect_text_rollout(
            model,
            reference,
            prompt_ids=prompt_ids,
            good_token_id=good_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            episodes=episodes,
            kl_coefficient=coefficient,
            generator=rollout_generator,
        )
        update = optimize_token_rollout(
            model,
            rollout,
            optimizer,
            epochs=epoch_count,
            minibatch_size=batch_size,
            clip_epsilon=0.2,
            gamma=0.95,
            gae_lambda=0.95,
            bootstrap_truncated=bootstrap_truncated,
            value_coefficient=value_weight,
            entropy_coefficient=entropy_weight,
            generator=optimizer_generator,
        )
        reports.append(
            {
                "iteration": iteration,
                "rollout_episode_count": episodes,
                "valid_action_count": int(rollout.valid_mask.sum()),
                "padding_transition_count": int((~rollout.valid_mask).sum()),
                "terminated_transition_count": int(rollout.terminated.sum()),
                "truncated_transition_count": int(rollout.truncated.sum()),
                "truncated_post_action_value_count": int(rollout.truncated.sum()),
                "sampled_task_reward_mean": float(
                    rollout.task_rewards.sum(dim=1).mean()
                ),
                "sampled_reference_log_ratio_mean": float(
                    rollout.sampled_reference_log_ratios[rollout.valid_mask].mean()
                ),
                "exact_categorical_kl_at_valid_states_mean": float(
                    rollout.exact_categorical_kls[rollout.valid_mask].mean()
                ),
                "old_log_probabilities_require_grad": (
                    rollout.old_log_probabilities.requires_grad
                ),
                **update,
            }
        )

    final_exact = exact_text_objectives(
        model,
        prompt_ids,
        good_token_id=good_token_id,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        max_context_length=max_context_length,
    )
    for name, parameter in reference.named_parameters():
        if not torch.equal(parameter.detach(), reference_before[name]):
            raise AssertionError(f"text PPO reference parameter changed: {name}")
    changed_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), initial_parameters[name])
    )
    return {
        "model_class": "separate GPT2 policy/value backbones+heads",
        "rendered_prompt": rendered_prompt,
        "prompt_token_ids": list(prompt_ids),
        "prompt_token_count": len(prompt_ids),
        "vocab_size": vocab_size,
        "good_token_id": good_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "initial_exact_objectives": asdict(initial_exact),
        "final_exact_objectives": asdict(final_exact),
        "initial_exact_expected_task_reward": initial_exact.expected_task_reward,
        "final_exact_expected_task_reward": final_exact.expected_task_reward,
        "initial_exact_good_then_eos_probability": (
            initial_exact.good_then_eos_probability
        ),
        "final_exact_good_then_eos_probability": (
            final_exact.good_then_eos_probability
        ),
        "first_sampled_task_reward_mean": reports[0]["sampled_task_reward_mean"],
        "last_sampled_task_reward_mean": reports[-1]["sampled_task_reward_mean"],
        "total_optimizer_steps": sum(
            int(report["optimizer_steps"]) for report in reports
        ),
        "bootstrap_truncated_in_optimizer": bootstrap_truncated,
        "finite_horizon_task_return_stops_at_generation_cap": True,
        "optimizer_matches_reported_finite_horizon_objective": (
            not bootstrap_truncated
        ),
        "reference_parameters_unchanged": True,
        "backbone_parameters_changed": any(
            name.startswith("backbone.") for name in changed_names
        ),
        "value_backbone_parameters_changed": any(
            name.startswith("value_backbone.") for name in changed_names
        ),
        "policy_head_parameters_changed": any(
            name.startswith("policy_head.") for name in changed_names
        ),
        "value_head_parameters_changed": any(
            name.startswith("value_head.") for name in changed_names
        ),
        "all_stored_old_log_probabilities_unchanged": all(
            report["stored_old_log_probabilities_unchanged"] for report in reports
        ),
        "maximum_snapshot_log_probability_error": max(
            float(report["snapshot_log_probability_max_error"])
            for report in reports
        ),
        "iterations": reports,
        "scope": {
            "device": "CPU",
            "local_wordlevel_tokenizer_executed": True,
            "chat_template_and_natural_language_prompt_executed": True,
            "random_tiny_gpt2_backbone_executed": True,
            "autoregressive_text_token_sampling_executed": True,
            "eos_termination_executed": True,
            "max_new_tokens_truncation_executed": True,
            "padding_mask_executed": True,
            "truncated_post_action_values_computed": True,
            "truncated_transition_value_bootstrap_executed": bootstrap_truncated,
            "finite_horizon_task_return_stops_at_generation_cap": True,
            "optimizer_matches_reported_finite_horizon_objective": (
                not bootstrap_truncated
            ),
            "exact_short_horizon_objective_enumerated": True,
            "learned_reward_model_executed": False,
            "human_preference_or_natural_language_quality_proved": False,
            "target_checkpoint_executed": False,
            "checkpoint_or_resume_executed": False,
            "cuda_or_distributed_execution": False,
        },
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
