"""Frozen learned-reward PPO counterexample with exhaustive proxy auditing."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
import torch.nn.functional as functional
from torch import Tensor
from transformers import GPT2Config, GPT2ForSequenceClassification

from about_llm.finetuning.ppo_text import (
    TEXT_CONTROL_HORIZON,
    batch_prompt_completions,
    build_text_contexts,
    build_text_control_tokenizer,
    collect_autoregressive_text_rollout,
    enumerate_stopped_responses,
    render_text_control_prompt,
)
from about_llm.finetuning.ppo_transformer import (
    TinyCausalActorCritic,
    finite_non_negative,
    freeze_copy,
    optimize_token_rollout,
    positive_integer,
)


@dataclass(frozen=True)
class ExactProxyObjectives:
    """Exact policy expectations over every reachable response."""

    probability_mass: float
    expected_centered_learned_reward: float
    authored_dense_task_reward: float
    authored_target_success_probability: float
    highest_rm_response_probability: float
    most_probable_response_ids: tuple[int, ...]
    most_probable_response_probability: float
    most_probable_response_centered_reward: float


@dataclass(frozen=True)
class LearnedRewardTable:
    """Frozen centered scores for all EOS/cap-stopped responses."""

    responses: tuple[tuple[int, ...], ...]
    centered_scores: Tensor
    one_token_scores: Tensor
    two_token_scores: Tensor
    target_response: tuple[int, int]
    rejected_response: tuple[int, int]
    highest_scoring_response: tuple[int, ...]
    target_rank: int
    centering_offset: float

    def score_rollout(self, actions: Tensor, valid_mask: Tensor) -> Tensor:
        rewards = torch.zeros(actions.shape, dtype=torch.float32)
        lengths = valid_mask.sum(dim=1)
        for row, length_tensor in enumerate(lengths):
            length = int(length_tensor)
            if length == 1:
                score = self.one_token_scores[actions[row, 0]]
            elif length == 2:
                score = self.two_token_scores[actions[row, 0], actions[row, 1]]
            else:
                raise AssertionError("two-token control produced an invalid response length")
            rewards[row, length - 1] = score
        return rewards


def _reward_scores(
    model: GPT2ForSequenceClassification,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    if logits.shape != (input_ids.shape[0], 1):
        raise AssertionError("reward model must emit one scalar per response")
    return cast(Tensor, logits[:, 0])


def _train_sparse_reward_model(
    *,
    prompt_ids: tuple[int, ...],
    vocab_size: int,
    good_token_id: int,
    bad_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    allowed_token_ids: tuple[int, ...],
    steps: int,
    seed: int,
) -> tuple[GPT2ForSequenceClassification, LearnedRewardTable, dict[str, Any]]:
    step_count = positive_integer(steps, "reward_model_steps")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("reward_model_seed must be a non-negative integer")
    max_context_length = len(prompt_ids) + TEXT_CONTROL_HORIZON
    target_response = (good_token_id, eos_token_id)
    rejected_response = (bad_token_id, eos_token_id)
    pair_ids, pair_mask = batch_prompt_completions(
        prompt_ids,
        (target_response, rejected_response),
        max_context_length=max_context_length,
        pad_token_id=pad_token_id,
    )
    torch.manual_seed(seed)
    config = GPT2Config(  # type: ignore[no-untyped-call]
        vocab_size=vocab_size,
        n_positions=max_context_length,
        n_ctx=max_context_length,
        n_embd=32,
        n_layer=1,
        n_head=2,
        num_labels=1,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        bos_token_id=eos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=False,
    )
    model = GPT2ForSequenceClassification(config)  # type: ignore[no-untyped-call]
    torch.nn.init.zeros_(model.score.weight)
    score_weight_before = model.score.weight.detach().clone()
    embedding_before = model.transformer.wte.weight.detach().clone()
    model.eval()
    with torch.inference_mode():
        initial_pair_scores = _reward_scores(model, pair_ids, pair_mask)
        initial_loss = float(
            functional.softplus(-(initial_pair_scores[0] - initial_pair_scores[1]))
        )
    if not math.isclose(initial_loss, math.log(2), abs_tol=1e-7):
        raise AssertionError("zero reward head must start at log(2) pairwise loss")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    model.train()
    for _ in range(step_count):
        optimizer.zero_grad(set_to_none=True)
        pair_scores = _reward_scores(model, pair_ids, pair_mask)
        loss = functional.softplus(-(pair_scores[0] - pair_scores[1]))
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    model.eval()
    responses = enumerate_stopped_responses(
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        horizon=TEXT_CONTROL_HORIZON,
        allowed_token_ids=allowed_token_ids,
    )
    response_ids, response_mask = batch_prompt_completions(
        prompt_ids,
        responses,
        max_context_length=max_context_length,
        pad_token_id=pad_token_id,
    )
    with torch.inference_mode():
        final_pair_scores = _reward_scores(model, pair_ids, pair_mask)
        final_margin = final_pair_scores[0] - final_pair_scores[1]
        final_loss = float(functional.softplus(-final_margin))
        centering_offset = float(final_pair_scores.mean())
        centered_scores = (
            _reward_scores(model, response_ids, response_mask) - centering_offset
        ).detach()
    target_index = responses.index(target_response)
    highest_index = int(torch.argmax(centered_scores))
    target_rank = int((centered_scores > centered_scores[target_index]).sum()) + 1
    one_token_scores = torch.full((vocab_size,), float("nan"))
    two_token_scores = torch.full((vocab_size, vocab_size), float("nan"))
    for response, score in zip(responses, centered_scores, strict=True):
        if len(response) == 1:
            one_token_scores[response[0]] = score
        else:
            two_token_scores[response[0], response[1]] = score
    table = LearnedRewardTable(
        responses=responses,
        centered_scores=centered_scores,
        one_token_scores=one_token_scores,
        two_token_scores=two_token_scores,
        target_response=target_response,
        rejected_response=rejected_response,
        highest_scoring_response=responses[highest_index],
        target_rank=target_rank,
        centering_offset=centering_offset,
    )
    score_head_changed = not torch.equal(model.score.weight.detach(), score_weight_before)
    embedding_changed = not torch.equal(
        model.transformer.wte.weight.detach(), embedding_before
    )
    if final_margin <= 0 or not score_head_changed or not embedding_changed:
        raise AssertionError("sparse reward-model pair did not train")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, table, {
        "training_pair_count": 1,
        "training_response_count": 2,
        "allowed_generation_token_count": len(allowed_token_ids),
        "reachable_response_count": len(responses),
        "reward_model_steps": step_count,
        "reward_model_seed": seed,
        "initial_pairwise_loss": initial_loss,
        "final_pairwise_loss": final_loss,
        "final_training_margin": float(final_margin),
        "strict_training_pair_accuracy": 1.0,
        "score_head_parameters_changed_during_rm_training": score_head_changed,
        "embedding_parameters_changed_during_rm_training": embedding_changed,
        "score_centering": "subtract final training-pair midpoint",
        "score_centering_offset": centering_offset,
        "target_centered_score": float(centered_scores[target_index]),
        "rejected_centered_score": float(
            centered_scores[responses.index(rejected_response)]
        ),
        "highest_centered_score": float(centered_scores[highest_index]),
        "highest_scoring_response_ids": list(responses[highest_index]),
        "target_response_rank_of_reachable": target_rank,
        "unseen_response_count": len(responses) - 2,
    }


@torch.inference_mode()
def exact_proxy_objectives(
    policy: TinyCausalActorCritic,
    table: LearnedRewardTable,
    prompt_ids: tuple[int, ...],
    *,
    pad_token_id: int,
    eos_token_id: int,
    vocab_size: int,
    allowed_token_ids: tuple[int, ...],
) -> ExactProxyObjectives:
    """Enumerate the complete stopped-response distribution under the policy."""

    policy.eval()
    if (eos_token_id,) not in table.responses:
        raise ValueError("reward table must contain the one-token EOS response")
    max_context_length = len(prompt_ids) + TEXT_CONTROL_HORIZON
    empty_prefix = torch.empty((1, 0), dtype=torch.long)
    first_context, first_mask = build_text_contexts(
        prompt_ids,
        empty_prefix,
        max_context_length=max_context_length,
        pad_token_id=pad_token_id,
    )
    first_logits, _ = policy.forward_contexts(first_context, first_mask)
    allowed_action_mask = torch.zeros(vocab_size, dtype=torch.bool)
    allowed_action_mask[list(allowed_token_ids)] = True
    first_logits = first_logits.masked_fill(
        ~allowed_action_mask.unsqueeze(0), float("-inf")
    )
    first_probabilities = torch.softmax(first_logits[0], dim=-1)
    first_actions = torch.arange(vocab_size, dtype=torch.long).unsqueeze(1)
    second_contexts, second_masks = build_text_contexts(
        prompt_ids,
        first_actions,
        max_context_length=max_context_length,
        pad_token_id=pad_token_id,
    )
    second_logits, _ = policy.forward_contexts(second_contexts, second_masks)
    second_logits = second_logits.masked_fill(
        ~allowed_action_mask.unsqueeze(0), float("-inf")
    )
    second_probabilities = torch.softmax(second_logits, dim=-1)
    response_probabilities: list[Tensor] = []
    for response in table.responses:
        if len(response) == 1:
            probability = first_probabilities[response[0]]
        else:
            probability = (
                first_probabilities[response[0]]
                * second_probabilities[response[0], response[1]]
            )
        response_probabilities.append(probability)
    probabilities = torch.stack(response_probabilities)
    target_index = table.responses.index(table.target_response)
    highest_index = table.responses.index(table.highest_scoring_response)
    most_probable_index = int(torch.argmax(probabilities))
    good_token_id = table.target_response[0]
    authored_dense_task_reward = first_probabilities[good_token_id]
    for first_token_id in range(vocab_size):
        if first_token_id != eos_token_id:
            authored_dense_task_reward = (
                authored_dense_task_reward
                + first_probabilities[first_token_id]
                * second_probabilities[first_token_id, eos_token_id]
            )
    return ExactProxyObjectives(
        probability_mass=float(probabilities.sum()),
        expected_centered_learned_reward=float(
            torch.sum(probabilities * table.centered_scores)
        ),
        authored_dense_task_reward=float(authored_dense_task_reward),
        authored_target_success_probability=float(probabilities[target_index]),
        highest_rm_response_probability=float(probabilities[highest_index]),
        most_probable_response_ids=table.responses[most_probable_index],
        most_probable_response_probability=float(probabilities[most_probable_index]),
        most_probable_response_centered_reward=float(
            table.centered_scores[most_probable_index]
        ),
    )


def run_smoke(
    *,
    reward_model_steps: int = 30,
    reward_model_seed: int = 17,
    ppo_iterations: int = 6,
    episodes_per_iteration: int = 128,
    epochs: int = 3,
    minibatch_size: int = 64,
    learning_rate: float = 0.01,
    kl_coefficient: float = 0.01,
) -> dict[str, Any]:
    """Train a sparse RM, freeze it, then show exact PPO proxy exploitation."""

    iteration_count = positive_integer(ppo_iterations, "ppo_iterations")
    episodes = positive_integer(episodes_per_iteration, "episodes_per_iteration")
    epoch_count = positive_integer(epochs, "epochs")
    batch_size = positive_integer(minibatch_size, "minibatch_size")
    rate = finite_non_negative(learning_rate, "learning_rate")
    if rate == 0:
        raise ValueError("learning_rate must be positive")
    coefficient = finite_non_negative(kl_coefficient, "kl_coefficient")
    tokenizer = build_text_control_tokenizer()
    rendered_prompt, prompt_ids = render_text_control_prompt(tokenizer)
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise AssertionError("tokenizer must define EOS and padding IDs")
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    good_token_id = tokenizer.convert_tokens_to_ids("good")
    bad_token_id = tokenizer.convert_tokens_to_ids("bad")
    vocab_size = len(tokenizer)
    allowed_token_texts = (
        "</s>",
        "Return",
        "one",
        "word.",
        "Say",
        "good.",
        "good",
        "bad",
    )
    allowed_token_ids = tuple(
        tokenizer.convert_tokens_to_ids(token) for token in allowed_token_texts
    )
    reward_model, reward_table, reward_report = _train_sparse_reward_model(
        prompt_ids=prompt_ids,
        vocab_size=vocab_size,
        good_token_id=good_token_id,
        bad_token_id=bad_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        allowed_token_ids=allowed_token_ids,
        steps=reward_model_steps,
        seed=reward_model_seed,
    )
    reward_before_ppo = {
        name: parameter.detach().clone()
        for name, parameter in reward_model.named_parameters()
    }

    torch.manual_seed(71)
    rollout_generator = torch.Generator(device="cpu").manual_seed(72)
    optimizer_generator = torch.Generator(device="cpu").manual_seed(73)
    max_context_length = len(prompt_ids) + TEXT_CONTROL_HORIZON
    policy = TinyCausalActorCritic(
        vocab_size=vocab_size,
        max_positions=max_context_length,
        bos_token_id=eos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        separate_value_backbone=True,
    ).to("cpu")
    reference = freeze_copy(policy)
    reference_before = {
        name: parameter.detach().clone()
        for name, parameter in reference.named_parameters()
    }
    optimizer = torch.optim.Adam(policy.parameters(), lr=rate)
    initial_exact = exact_proxy_objectives(
        policy,
        reward_table,
        prompt_ids,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        allowed_token_ids=allowed_token_ids,
    )
    reports: list[dict[str, Any]] = []
    for iteration in range(iteration_count):
        rollout = collect_autoregressive_text_rollout(
            policy,
            reference,
            prompt_ids=prompt_ids,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            horizon=TEXT_CONTROL_HORIZON,
            episodes=episodes,
            kl_coefficient=coefficient,
            task_reward_fn=reward_table.score_rollout,
            generator=rollout_generator,
            allowed_token_ids=allowed_token_ids,
        )
        update = optimize_token_rollout(
            policy,
            rollout,
            optimizer,
            epochs=epoch_count,
            minibatch_size=batch_size,
            clip_epsilon=0.2,
            gamma=1.0,
            gae_lambda=0.95,
            bootstrap_truncated=False,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            generator=optimizer_generator,
        )
        exact_after_iteration = exact_proxy_objectives(
            policy,
            reward_table,
            prompt_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            allowed_token_ids=allowed_token_ids,
        )
        reports.append(
            {
                "iteration": iteration,
                "sampled_centered_learned_reward_mean": float(
                    rollout.task_rewards.sum(dim=1).mean()
                ),
                "terminated_transition_count": int(rollout.terminated.sum()),
                "truncated_transition_count": int(rollout.truncated.sum()),
                "padding_transition_count": int((~rollout.valid_mask).sum()),
                "exact_after_iteration": asdict(exact_after_iteration),
                **update,
            }
        )

    final_exact = exact_proxy_objectives(
        policy,
        reward_table,
        prompt_ids,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        allowed_token_ids=allowed_token_ids,
    )
    reward_unchanged = all(
        torch.equal(parameter.detach(), reward_before_ppo[name])
        for name, parameter in reward_model.named_parameters()
    )
    reference_unchanged = all(
        torch.equal(parameter.detach(), reference_before[name])
        for name, parameter in reference.named_parameters()
    )
    if not reward_unchanged or not reference_unchanged:
        raise AssertionError("frozen reward/reference model changed during PPO")
    proxy_improved = (
        final_exact.expected_centered_learned_reward
        > initial_exact.expected_centered_learned_reward
    )
    authored_success_improved = (
        final_exact.authored_target_success_probability
        > initial_exact.authored_target_success_probability
    )
    authored_dense_reward_improved = (
        final_exact.authored_dense_task_reward
        > initial_exact.authored_dense_task_reward
    )
    reward_hacking_counterexample = proxy_improved and not authored_success_improved
    return {
        "rendered_prompt": rendered_prompt,
        "prompt_token_ids": list(prompt_ids),
        "vocab_size": vocab_size,
        "good_token_id": good_token_id,
        "bad_token_id": bad_token_id,
        "eos_token_id": eos_token_id,
        "allowed_generation_token_ids": list(allowed_token_ids),
        "allowed_generation_tokens": list(allowed_token_texts),
        "reward_model": reward_report,
        "highest_scoring_response_tokens": tokenizer.convert_ids_to_tokens(
            list(reward_table.highest_scoring_response)
        ),
        "target_response_ids": list(reward_table.target_response),
        "initial_exact_objectives": asdict(initial_exact),
        "final_exact_objectives": asdict(final_exact),
        "exact_proxy_reward_improved": proxy_improved,
        "exact_authored_dense_task_reward_improved": authored_dense_reward_improved,
        "exact_authored_target_success_improved": authored_success_improved,
        "reward_hacking_counterexample_observed": reward_hacking_counterexample,
        "reward_hacking_counterexample_definition": (
            "exact learned proxy improves while strict authored target success declines"
        ),
        "reward_model_parameters_unchanged_during_ppo": reward_unchanged,
        "reference_parameters_unchanged_during_ppo": reference_unchanged,
        "total_ppo_optimizer_steps": sum(
            int(report["optimizer_steps"]) for report in reports
        ),
        "all_stored_old_log_probabilities_unchanged": all(
            bool(report["stored_old_log_probabilities_unchanged"])
            for report in reports
        ),
        "maximum_snapshot_log_probability_error": max(
            float(report["snapshot_log_probability_max_error"])
            for report in reports
        ),
        "iterations": reports,
        "scope": {
            "device": "CPU",
            "local_wordlevel_tokenizer_and_chat_template_executed": True,
            "generation_allowlist_bound_to_sampling_and_ppo_distribution": True,
            "pairwise_transformer_reward_model_optimizer_executed": True,
            "sparse_authored_preference_pair_not_human_labels": True,
            "frozen_learned_sequence_reward_bound_to_terminal_action": True,
            "all_reachable_two_token_responses_enumerated": True,
            "ppo_optimizer_executed_against_learned_proxy": True,
            "exact_proxy_reward_improved": proxy_improved,
            "exact_authored_dense_task_reward_improved": (
                authored_dense_reward_improved
            ),
            "exact_authored_target_success_improved": authored_success_improved,
            "controlled_reward_hacking_counterexample_observed": (
                reward_hacking_counterexample
            ),
            "reward_model_quality_or_robustness_proved": False,
            "human_preference_or_natural_language_quality_proved": False,
            "target_checkpoint_executed": False,
            "cuda_or_distributed_execution": False,
            "production_ppo_stability_proved": False,
        },
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
