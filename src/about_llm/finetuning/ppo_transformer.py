"""Optional PyTorch/Transformers primitives for tiny causal-policy PPO controls."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers import GPT2Config, GPT2Model

from about_llm.finetuning.ppo import generalized_advantage_estimation


class TinyCausalActorCritic(nn.Module):
    """Random tiny GPT-2 backbone with categorical policy and scalar value heads."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_positions: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        embedding_dim: int = 16,
        num_layers: int = 1,
        num_heads: int = 2,
        separate_value_backbone: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(separate_value_backbone, bool):
            raise ValueError("separate_value_backbone must be boolean")
        sizes = {
            "vocab_size": vocab_size,
            "max_positions": max_positions,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
        }
        if any(isinstance(value, bool) or value <= 0 for value in sizes.values()):
            raise ValueError(f"all Transformer sizes must be positive integers: {sizes}")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        for label, token_id in (
            ("bos_token_id", bos_token_id),
            ("eos_token_id", eos_token_id),
            ("pad_token_id", pad_token_id),
        ):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < vocab_size
            ):
                raise ValueError(f"{label} must be an integer inside the vocabulary")
        config = GPT2Config(  # type: ignore[no-untyped-call]
            vocab_size=vocab_size,
            n_positions=max_positions,
            n_ctx=max_positions,
            n_embd=embedding_dim,
            n_layer=num_layers,
            n_head=num_heads,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=False,
        )
        self.backbone = GPT2Model(config)  # type: ignore[no-untyped-call]
        self.value_backbone = (
            GPT2Model(config)  # type: ignore[no-untyped-call]
            if separate_value_backbone
            else None
        )
        self.policy_head = nn.Linear(embedding_dim, vocab_size, bias=True)
        self.value_head = nn.Linear(embedding_dim, 1, bias=True)
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward_contexts(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must share [batch, time] shape")
        if input_ids.shape[1] == 0:
            raise ValueError("contexts must contain at least one token position")
        if attention_mask.dtype == torch.bool:
            valid_counts = attention_mask.sum(dim=1)
        elif attention_mask.dtype in (torch.int32, torch.int64):
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("attention_mask must be binary")
            valid_counts = attention_mask.sum(dim=1)
        else:
            raise ValueError("attention_mask must be boolean or integer")
        if torch.any(valid_counts <= 0):
            raise ValueError("every context must select at least one token")
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        final_indices = valid_counts - 1
        last_hidden_state = cast(Tensor, outputs.last_hidden_state)
        policy_hidden = last_hidden_state[
            torch.arange(input_ids.shape[0], device=input_ids.device), final_indices
        ]
        value_hidden = policy_hidden
        if self.value_backbone is not None:
            value_outputs = self.value_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            value_last_hidden_state = cast(Tensor, value_outputs.last_hidden_state)
            value_hidden = value_last_hidden_state[
                torch.arange(input_ids.shape[0], device=input_ids.device), final_indices
            ]
        return (
            self.policy_head(policy_hidden),
            self.value_head(value_hidden).squeeze(-1),
        )


@dataclass(frozen=True)
class TransformerRollout:
    contexts: Tensor
    attention_masks: Tensor
    actions: Tensor
    task_rewards: Tensor
    shaped_rewards: Tensor
    sampled_reference_log_ratios: Tensor
    old_log_probabilities: Tensor
    reference_log_probabilities: Tensor
    old_values: Tensor
    next_values: Tensor
    valid_mask: Tensor
    terminated: Tensor
    truncated: Tensor
    exact_categorical_kls: Tensor
    behavior_policy_snapshot: TinyCausalActorCritic
    allowed_action_mask: Tensor | None = None


def positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def finite_non_negative(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def freeze_copy(model: TinyCausalActorCritic) -> TinyCausalActorCritic:
    frozen = copy.deepcopy(model).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def _advantages_and_returns(
    rollout: TransformerRollout,
    *,
    gamma: float,
    gae_lambda: float,
    bootstrap_truncated: bool,
) -> tuple[Tensor, Tensor]:
    estimate = generalized_advantage_estimation(
        rollout.shaped_rewards.numpy(),
        rollout.old_values.numpy(),
        rollout.next_values.numpy(),
        valid_mask=rollout.valid_mask.numpy(),
        terminated=rollout.terminated.numpy(),
        truncated=rollout.truncated.numpy(),
        gamma=gamma,
        gae_lambda=gae_lambda,
        bootstrap_truncated=bootstrap_truncated,
    )
    return (
        torch.from_numpy(estimate.advantages).to(torch.float32),
        torch.from_numpy(estimate.returns).to(torch.float32),
    )


def _flat_policy_values(
    model: TinyCausalActorCritic,
    contexts: Tensor,
    attention_masks: Tensor,
    actions: Tensor,
    allowed_action_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    logits, values = model.forward_contexts(contexts, attention_masks)
    if allowed_action_mask is not None:
        if (
            allowed_action_mask.ndim != 1
            or allowed_action_mask.shape[0] != logits.shape[1]
            or allowed_action_mask.dtype != torch.bool
            or not torch.any(allowed_action_mask)
        ):
            raise ValueError("allowed_action_mask must select vocabulary entries")
        allowed_log_probabilities = torch.log_softmax(
            logits[:, allowed_action_mask], dim=-1
        )
        all_log_probabilities = torch.full_like(logits, float("-inf"))
        all_log_probabilities[:, allowed_action_mask] = allowed_log_probabilities
    else:
        all_log_probabilities = torch.log_softmax(logits, dim=-1)
    action_log_probabilities = all_log_probabilities.gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)
    return all_log_probabilities, action_log_probabilities, values


def optimize_token_rollout(
    model: TinyCausalActorCritic,
    rollout: TransformerRollout,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    gamma: float,
    gae_lambda: float,
    bootstrap_truncated: bool,
    value_coefficient: float,
    entropy_coefficient: float,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Optimize a fixed rollout without recomputing old policy statistics."""

    epoch_count = positive_integer(epochs, "epochs")
    batch_size = positive_integer(minibatch_size, "minibatch_size")
    epsilon = finite_non_negative(clip_epsilon, "clip_epsilon")
    if epsilon == 0 or epsilon > 1:
        raise ValueError("clip_epsilon must be in (0, 1]")
    value_weight = finite_non_negative(value_coefficient, "value_coefficient")
    entropy_weight = finite_non_negative(entropy_coefficient, "entropy_coefficient")
    allowed_action_mask = (
        None
        if rollout.allowed_action_mask is None
        else rollout.allowed_action_mask.clone()
    )
    advantages, returns = _advantages_and_returns(
        rollout,
        gamma=gamma,
        gae_lambda=gae_lambda,
        bootstrap_truncated=bootstrap_truncated,
    )
    active_advantages = advantages[rollout.valid_mask]
    if active_advantages.numel() == 0:
        raise ValueError("rollout valid_mask must select at least one action")
    advantage_mean = active_advantages.mean()
    advantage_std = active_advantages.std(unbiased=False)
    normalized_advantages = torch.zeros_like(advantages)
    normalized_advantages[rollout.valid_mask] = (
        active_advantages - advantage_mean
    ) / torch.clamp(advantage_std, min=1e-8)

    flat_contexts = rollout.contexts[rollout.valid_mask]
    flat_masks = rollout.attention_masks[rollout.valid_mask]
    flat_actions = rollout.actions[rollout.valid_mask]
    flat_old_log_probs = rollout.old_log_probabilities[rollout.valid_mask]
    flat_advantages = normalized_advantages[rollout.valid_mask]
    flat_returns = returns[rollout.valid_mask]
    old_log_probs_before = rollout.old_log_probabilities.clone()
    optimizer_steps = 0
    losses: list[dict[str, float]] = []
    model.train()
    for _ in range(epoch_count):
        permutation = torch.randperm(flat_actions.numel(), generator=generator)
        for start in range(0, flat_actions.numel(), batch_size):
            indices = permutation[start : start + batch_size]
            all_log_probs, new_log_probs, predicted_values = _flat_policy_values(
                model,
                flat_contexts[indices],
                flat_masks[indices],
                flat_actions[indices],
                allowed_action_mask,
            )
            ratios = torch.exp(new_log_probs - flat_old_log_probs[indices])
            batch_advantages = flat_advantages[indices]
            unclipped = ratios * batch_advantages
            clipped = torch.clamp(
                ratios, 1 - epsilon, 1 + epsilon
            ) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * torch.mean(
                (predicted_values - flat_returns[indices]) ** 2
            )
            entropy_log_probs = (
                all_log_probs
                if allowed_action_mask is None
                else all_log_probs[:, allowed_action_mask]
            )
            probabilities = torch.exp(entropy_log_probs)
            entropy = -torch.mean(
                torch.sum(probabilities * entropy_log_probs, dim=-1)
            )
            total_loss = (
                policy_loss + value_weight * value_loss - entropy_weight * entropy
            )
            if not torch.isfinite(total_loss):
                raise AssertionError("Transformer PPO loss must remain finite")
            optimizer.zero_grad()
            total_loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            optimizer_steps += 1
            losses.append(
                {
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "total_loss": float(total_loss.detach()),
                }
            )

    model.eval()
    with torch.inference_mode():
        _, snapshot_log_probs, _ = _flat_policy_values(
            rollout.behavior_policy_snapshot,
            flat_contexts,
            flat_masks,
            flat_actions,
            allowed_action_mask,
        )
        _, current_log_probs, _ = _flat_policy_values(
            model,
            flat_contexts,
            flat_masks,
            flat_actions,
            allowed_action_mask,
        )
        ratios = torch.exp(current_log_probs - flat_old_log_probs)
        approximate_kl = torch.expm1(
            current_log_probs - flat_old_log_probs
        ) - (current_log_probs - flat_old_log_probs)
    snapshot_error = float(torch.max(torch.abs(snapshot_log_probs - flat_old_log_probs)))
    if snapshot_error > 1e-7:
        raise AssertionError("old log-probs do not match the behavior snapshot")
    if not torch.equal(rollout.old_log_probabilities, old_log_probs_before):
        raise AssertionError("stored Transformer old log-probs were mutated")
    return {
        "optimizer_steps": optimizer_steps,
        "initial_minibatch_losses": losses[0],
        "final_minibatch_losses": losses[-1],
        "advantage_mean_before_normalization": float(advantage_mean),
        "advantage_std_before_normalization": float(advantage_std),
        "stored_old_log_probabilities_unchanged": True,
        "snapshot_log_probability_max_error": snapshot_error,
        "post_update_clip_fraction": float(
            ((ratios < 1 - epsilon) | (ratios > 1 + epsilon))
            .to(torch.float32)
            .mean()
        ),
        "post_update_minimum_ratio": float(ratios.min()),
        "post_update_maximum_ratio": float(ratios.max()),
        "post_update_approximate_sampled_kl": float(approximate_kl.mean()),
    }
