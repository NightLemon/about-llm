"""Local text-tokenizer and variable-length rollout helpers for PPO controls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import torch
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import WhitespaceSplit  # type: ignore[import-untyped]
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from about_llm.finetuning.ppo_transformer import (
    TinyCausalActorCritic,
    TransformerRollout,
    finite_non_negative,
    freeze_copy,
    positive_integer,
)

TEXT_CONTROL_HORIZON = 2
TEXT_CONTROL_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|> ' + message['content'] + ' ' + eos_token + ' ' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}"
)
TEXT_CONTROL_MESSAGES = (
    {"role": "system", "content": "Return one word."},
    {"role": "user", "content": "Say good."},
)

TextRewardFunction = Callable[[Tensor, Tensor], Tensor]


def build_text_control_tokenizer() -> PreTrainedTokenizerFast:
    """Build the fully local tokenizer shared by text PPO controls."""

    vocabulary = {
        "[UNK]": 0,
        "[PAD]": 1,
        "</s>": 2,
        "<|system|>": 3,
        "<|user|>": 4,
        "<|assistant|>": 5,
        "Return": 6,
        "one": 7,
        "word.": 8,
        "Say": 9,
        "good.": 10,
        "good": 11,
        "bad": 12,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="</s>",
        eos_token="</s>",
    )
    tokenizer.chat_template = TEXT_CONTROL_CHAT_TEMPLATE
    return tokenizer


def render_text_control_prompt(
    tokenizer: PreTrainedTokenizerFast,
) -> tuple[str, tuple[int, ...]]:
    """Render the control prompt as both text and integer token IDs."""

    rendered = tokenizer.apply_chat_template(
        list(TEXT_CONTROL_MESSAGES), tokenize=False, add_generation_prompt=True
    )
    token_ids = tokenizer.apply_chat_template(
        list(TEXT_CONTROL_MESSAGES), tokenize=True, add_generation_prompt=True
    )
    if not isinstance(rendered, str) or not rendered:
        raise AssertionError("chat template must render non-empty text")
    if not isinstance(token_ids, list) or not token_ids:
        raise AssertionError("chat template must produce non-empty token IDs")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids):
        raise AssertionError("chat template token IDs must be integers")
    return rendered, tuple(cast(list[int], token_ids))


def build_text_contexts(
    prompt_ids: tuple[int, ...],
    generated_prefixes: Tensor,
    *,
    max_context_length: int,
    pad_token_id: int,
) -> tuple[Tensor, Tensor]:
    """Right-pad prompt plus generated prefixes for next-token policy/value forward."""

    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if generated_prefixes.ndim != 2 or generated_prefixes.dtype != torch.long:
        raise ValueError("generated_prefixes must be a rank-2 torch.long tensor")
    if isinstance(max_context_length, bool) or max_context_length <= 0:
        raise ValueError("max_context_length must be positive")
    batch = generated_prefixes.shape[0]
    prefix_length = generated_prefixes.shape[1]
    used_length = len(prompt_ids) + prefix_length
    if used_length > max_context_length:
        raise ValueError("prompt and generated prefix exceed max_context_length")
    contexts = torch.full(
        (batch, max_context_length), pad_token_id, dtype=torch.long
    )
    attention_masks = torch.zeros_like(contexts)
    prompt = torch.tensor(prompt_ids, dtype=torch.long)
    contexts[:, : len(prompt_ids)] = prompt
    attention_masks[:, : len(prompt_ids)] = 1
    if prefix_length:
        contexts[:, len(prompt_ids) : used_length] = generated_prefixes
        attention_masks[:, len(prompt_ids) : used_length] = 1
    return contexts, attention_masks


def batch_prompt_completions(
    prompt_ids: tuple[int, ...],
    completions: Sequence[Sequence[int]],
    *,
    max_context_length: int,
    pad_token_id: int,
) -> tuple[Tensor, Tensor]:
    """Batch complete prompt/response sequences without silent truncation."""

    if not completions:
        raise ValueError("completions must not be empty")
    input_ids = torch.full(
        (len(completions), max_context_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, completion in enumerate(completions):
        sequence = prompt_ids + tuple(completion)
        if len(sequence) > max_context_length:
            raise ValueError("prompt and completion exceed max_context_length")
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = 1
    return input_ids, attention_mask


def enumerate_stopped_responses(
    *,
    vocab_size: int,
    eos_token_id: int,
    horizon: int,
    allowed_token_ids: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate responses that stop at EOS or at the finite generation cap."""

    size = positive_integer(vocab_size, "vocab_size")
    steps = positive_integer(horizon, "horizon")
    if not 0 <= eos_token_id < size:
        raise ValueError("eos_token_id must be inside the vocabulary")
    allowed = (
        tuple(range(size))
        if allowed_token_ids is None
        else tuple(allowed_token_ids)
    )
    if (
        not allowed
        or len(set(allowed)) != len(allowed)
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < size
            for token_id in allowed
        )
    ):
        raise ValueError("allowed_token_ids must be unique vocabulary IDs")
    if eos_token_id not in allowed:
        raise ValueError("allowed_token_ids must include eos_token_id")
    responses: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...]) -> None:
        if prefix and (prefix[-1] == eos_token_id or len(prefix) == steps):
            responses.append(prefix)
            return
        for token_id in allowed:
            visit((*prefix, token_id))

    visit(())
    return tuple(responses)


@torch.inference_mode()
def collect_autoregressive_text_rollout(
    model: TinyCausalActorCritic,
    reference: TinyCausalActorCritic,
    *,
    prompt_ids: tuple[int, ...],
    eos_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    horizon: int,
    episodes: int,
    kl_coefficient: float,
    task_reward_fn: TextRewardFunction,
    generator: torch.Generator,
    allowed_token_ids: Sequence[int] | None = None,
) -> TransformerRollout:
    """Collect variable-length text actions and bind externally defined rewards."""

    episode_count = positive_integer(episodes, "episodes")
    step_count = positive_integer(horizon, "horizon")
    coefficient = finite_non_negative(kl_coefficient, "kl_coefficient")
    if not 0 <= eos_token_id < vocab_size or not 0 <= pad_token_id < vocab_size:
        raise ValueError("EOS and padding IDs must be inside the vocabulary")
    allowed = (
        tuple(range(vocab_size))
        if allowed_token_ids is None
        else tuple(allowed_token_ids)
    )
    if (
        not allowed
        or len(set(allowed)) != len(allowed)
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocab_size
            for token_id in allowed
        )
    ):
        raise ValueError("allowed_token_ids must be unique vocabulary IDs")
    if eos_token_id not in allowed:
        raise ValueError("allowed_token_ids must include eos_token_id")
    allowed_ids = torch.tensor(allowed, dtype=torch.long)
    allowed_action_mask = torch.zeros(vocab_size, dtype=torch.bool)
    allowed_action_mask[allowed_ids] = True
    max_context_length = len(prompt_ids) + step_count
    behavior = freeze_copy(model)
    contexts = torch.full(
        (episode_count, step_count, max_context_length),
        pad_token_id,
        dtype=torch.long,
    )
    attention_masks = torch.zeros_like(contexts)
    actions = torch.full(
        (episode_count, step_count), pad_token_id, dtype=torch.long
    )
    old_log_probabilities = torch.zeros(
        (episode_count, step_count), dtype=torch.float32
    )
    reference_log_probabilities = torch.zeros_like(old_log_probabilities)
    old_values = torch.zeros_like(old_log_probabilities)
    exact_categorical_kls = torch.zeros_like(old_log_probabilities)
    valid_mask = torch.zeros_like(actions, dtype=torch.bool)
    terminated = torch.zeros_like(actions, dtype=torch.bool)
    truncated = torch.zeros_like(actions, dtype=torch.bool)
    active = torch.ones(episode_count, dtype=torch.bool)

    for step in range(step_count):
        active_indices = torch.nonzero(active, as_tuple=False).squeeze(1)
        if active_indices.numel() == 0:
            break
        generated_prefixes = actions[active_indices, :step]
        step_contexts, step_masks = build_text_contexts(
            prompt_ids,
            generated_prefixes,
            max_context_length=max_context_length,
            pad_token_id=pad_token_id,
        )
        contexts[active_indices, step] = step_contexts
        attention_masks[active_indices, step] = step_masks
        behavior_logits, behavior_values = behavior.forward_contexts(
            step_contexts, step_masks
        )
        reference_logits, _ = reference.forward_contexts(step_contexts, step_masks)
        behavior_log_probs = torch.log_softmax(
            behavior_logits[:, allowed_ids], dim=-1
        )
        reference_log_probs = torch.log_softmax(
            reference_logits[:, allowed_ids], dim=-1
        )
        behavior_probabilities = torch.exp(behavior_log_probs)
        local_actions = torch.multinomial(
            behavior_probabilities, 1, replacement=True, generator=generator
        ).squeeze(1)
        step_actions = allowed_ids[local_actions]
        actions[active_indices, step] = step_actions
        valid_mask[active_indices, step] = True
        old_log_probabilities[active_indices, step] = behavior_log_probs.gather(
            1, local_actions.unsqueeze(1)
        ).squeeze(1)
        reference_log_probabilities[active_indices, step] = reference_log_probs.gather(
            1, local_actions.unsqueeze(1)
        ).squeeze(1)
        old_values[active_indices, step] = behavior_values
        exact_categorical_kls[active_indices, step] = torch.sum(
            behavior_probabilities * (behavior_log_probs - reference_log_probs),
            dim=-1,
        )
        ended = step_actions == eos_token_id
        terminated[active_indices[ended], step] = True
        if step == step_count - 1:
            truncated[active_indices[~ended], step] = True
        active[active_indices[ended]] = False

    task_rewards = task_reward_fn(actions.clone(), valid_mask.clone()).to(torch.float32)
    if task_rewards.shape != old_values.shape:
        raise ValueError("task_reward_fn must return one reward per rollout slot")
    if not torch.all(torch.isfinite(task_rewards)):
        raise ValueError("task_reward_fn returned a non-finite reward")
    if torch.any(task_rewards[~valid_mask] != 0):
        raise ValueError("task_reward_fn must assign zero reward to padding")
    sampled_reference_log_ratios = old_log_probabilities - reference_log_probabilities
    sampled_reference_log_ratios[~valid_mask] = 0
    shaped_rewards = task_rewards - coefficient * sampled_reference_log_ratios
    shaped_rewards[~valid_mask] = 0
    next_values = torch.zeros_like(old_values)
    next_values[:, :-1] = torch.where(
        valid_mask[:, 1:], old_values[:, 1:], torch.zeros_like(old_values[:, 1:])
    )
    truncated_indices = torch.nonzero(
        truncated[:, step_count - 1], as_tuple=False
    ).squeeze(1)
    if truncated_indices.numel():
        post_actions = actions[truncated_indices, :step_count]
        post_contexts, post_masks = build_text_contexts(
            prompt_ids,
            post_actions,
            max_context_length=max_context_length,
            pad_token_id=pad_token_id,
        )
        _, post_values = behavior.forward_contexts(post_contexts, post_masks)
        next_values[truncated_indices, step_count - 1] = post_values

    if torch.any((terminated | truncated) & ~valid_mask):
        raise AssertionError("padding transitions cannot carry episode boundaries")
    if torch.any(valid_mask[:, 1:] & ~valid_mask[:, :-1]):
        raise AssertionError("a later action cannot exist after padding")
    return TransformerRollout(
        contexts=contexts,
        attention_masks=attention_masks,
        actions=actions,
        task_rewards=task_rewards,
        shaped_rewards=shaped_rewards,
        sampled_reference_log_ratios=sampled_reference_log_ratios,
        old_log_probabilities=old_log_probabilities,
        reference_log_probabilities=reference_log_probabilities,
        old_values=old_values,
        next_values=next_values,
        valid_mask=valid_mask,
        terminated=terminated,
        truncated=truncated,
        exact_categorical_kls=exact_categorical_kls,
        behavior_policy_snapshot=behavior,
        allowed_action_mask=(
            None if allowed_token_ids is None else allowed_action_mask
        ),
    )
