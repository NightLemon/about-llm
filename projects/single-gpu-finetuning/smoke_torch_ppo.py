"""Deterministic CPU PPO control on an authored two-state environment."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn

from about_llm.finetuning import generalized_advantage_estimation


@dataclass(frozen=True)
class RolloutBatch:
    states: Tensor
    actions: Tensor
    rewards: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    next_values: Tensor
    valid_mask: Tensor
    terminated: Tensor
    truncated: Tensor
    policy_logits_snapshot: Tensor


class TinyActorCritic(nn.Module):
    """Tabular categorical policy and scalar value for two observable states."""

    def __init__(self) -> None:
        super().__init__()
        self.policy_logits = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
        self.state_values = nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def policy_log_probabilities(self, states: Tensor) -> Tensor:
        return torch.log_softmax(self.policy_logits[states], dim=-1)

    def values(self, states: Tensor) -> Tensor:
        return self.state_values[states]


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_positive(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


@torch.inference_mode()
def exact_expected_return(model: TinyActorCritic) -> float:
    """Return exact undiscounted reward for the fixed two-step environment."""

    probabilities = torch.softmax(model.policy_logits, dim=-1)
    return float(probabilities[0, 1] + probabilities[1, 0])


@torch.inference_mode()
def collect_rollout(
    model: TinyActorCritic,
    *,
    episodes: int,
    generator: torch.Generator,
) -> RolloutBatch:
    """Sample full two-step episodes and freeze behavior-policy statistics."""

    episode_count = _positive_integer(episodes, "episodes")
    states = torch.tensor([0, 1], dtype=torch.long).repeat(episode_count, 1)
    policy_logits_snapshot = model.policy_logits.detach().clone()
    log_probabilities = torch.log_softmax(policy_logits_snapshot, dim=-1)
    probabilities = torch.exp(log_probabilities)
    first_actions = torch.multinomial(
        probabilities[0], episode_count, replacement=True, generator=generator
    )
    second_actions = torch.multinomial(
        probabilities[1], episode_count, replacement=True, generator=generator
    )
    actions = torch.stack((first_actions, second_actions), dim=1)
    old_log_probabilities = log_probabilities[states, actions].clone()
    rewards = torch.stack(
        ((first_actions == 1).to(torch.float64), (second_actions == 0).to(torch.float64)),
        dim=1,
    )
    old_values = model.values(states).detach().clone()
    next_values = torch.zeros_like(old_values)
    next_values[:, 0] = model.state_values[1].detach()
    valid_mask = torch.ones_like(actions, dtype=torch.bool)
    terminated = torch.zeros_like(actions, dtype=torch.bool)
    terminated[:, 1] = True
    truncated = torch.zeros_like(actions, dtype=torch.bool)
    return RolloutBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        old_log_probabilities=old_log_probabilities,
        old_values=old_values,
        next_values=next_values,
        valid_mask=valid_mask,
        terminated=terminated,
        truncated=truncated,
        policy_logits_snapshot=policy_logits_snapshot,
    )


def _advantages_and_returns(
    rollout: RolloutBatch,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    estimate = generalized_advantage_estimation(
        rollout.rewards.numpy(),
        rollout.old_values.numpy(),
        rollout.next_values.numpy(),
        valid_mask=rollout.valid_mask.numpy(),
        terminated=rollout.terminated.numpy(),
        truncated=rollout.truncated.numpy(),
        gamma=gamma,
        gae_lambda=gae_lambda,
        bootstrap_truncated=True,
    )
    advantages = torch.from_numpy(estimate.advantages).to(torch.float64)
    returns = torch.from_numpy(estimate.returns).to(torch.float64)
    return advantages, returns


def _evaluate_rollout_objective(
    model: TinyActorCritic,
    rollout: RolloutBatch,
    normalized_advantages: Tensor,
    *,
    clip_epsilon: float,
) -> dict[str, float]:
    with torch.inference_mode():
        new_log_probabilities = model.policy_log_probabilities(rollout.states)[
            rollout.valid_mask
        ].gather(1, rollout.actions[rollout.valid_mask].unsqueeze(1)).squeeze(1)
        old_log_probabilities = rollout.old_log_probabilities[rollout.valid_mask]
        ratios = torch.exp(new_log_probabilities - old_log_probabilities)
        active_advantages = normalized_advantages[rollout.valid_mask]
        unclipped = ratios * active_advantages
        clipped = torch.clamp(
            ratios, 1 - clip_epsilon, 1 + clip_epsilon
        ) * active_advantages
        surrogate = torch.minimum(unclipped, clipped)
        approximate_kl = torch.expm1(
            new_log_probabilities - old_log_probabilities
        ) - (new_log_probabilities - old_log_probabilities)
        return {
            "surrogate_objective": float(surrogate.mean()),
            "clip_fraction": float(
                ((ratios < 1 - clip_epsilon) | (ratios > 1 + clip_epsilon))
                .to(torch.float64)
                .mean()
            ),
            "approximate_sampled_kl": float(approximate_kl.mean()),
            "maximum_probability_ratio": float(ratios.max()),
            "minimum_probability_ratio": float(ratios.min()),
        }


def optimize_rollout(
    model: TinyActorCritic,
    rollout: RolloutBatch,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    gamma: float,
    gae_lambda: float,
    value_coefficient: float,
    entropy_coefficient: float,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Run multiple PPO epochs while keeping rollout targets and old log-probs fixed."""

    epoch_count = _positive_integer(epochs, "epochs")
    batch_size = _positive_integer(minibatch_size, "minibatch_size")
    advantages, returns = _advantages_and_returns(
        rollout, gamma=gamma, gae_lambda=gae_lambda
    )
    active_advantages = advantages[rollout.valid_mask]
    advantage_mean = active_advantages.mean()
    advantage_std = active_advantages.std(unbiased=False)
    normalized_advantages = torch.zeros_like(advantages)
    normalized_advantages[rollout.valid_mask] = (
        active_advantages - advantage_mean
    ) / torch.clamp(advantage_std, min=1e-8)

    flat_states = rollout.states[rollout.valid_mask]
    flat_actions = rollout.actions[rollout.valid_mask]
    flat_old_log_probs = rollout.old_log_probabilities[rollout.valid_mask]
    flat_advantages = normalized_advantages[rollout.valid_mask]
    flat_returns = returns[rollout.valid_mask]
    old_log_probs_before = rollout.old_log_probabilities.clone()
    loss_rows: list[dict[str, float]] = []
    optimizer_steps = 0
    for _ in range(epoch_count):
        permutation = torch.randperm(
            flat_states.numel(), generator=generator
        )
        for start in range(0, flat_states.numel(), batch_size):
            indices = permutation[start : start + batch_size]
            states = flat_states[indices]
            actions = flat_actions[indices]
            old_log_probs = flat_old_log_probs[indices]
            batch_advantages = flat_advantages[indices]
            target_returns = flat_returns[indices]

            all_log_probs = model.policy_log_probabilities(states)
            new_log_probs = all_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratios = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratios * batch_advantages
            clipped = torch.clamp(
                ratios, 1 - clip_epsilon, 1 + clip_epsilon
            ) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            predicted_values = model.values(states)
            value_loss = 0.5 * torch.mean((predicted_values - target_returns) ** 2)
            probabilities = torch.exp(all_log_probs)
            entropy = -torch.mean(torch.sum(probabilities * all_log_probs, dim=-1))
            total_loss = (
                policy_loss
                + value_coefficient * value_loss
                - entropy_coefficient * entropy
            )
            if not torch.isfinite(total_loss):
                raise AssertionError("PPO minibatch loss must remain finite")
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer_steps += 1
            loss_rows.append(
                {
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "total_loss": float(total_loss.detach()),
                }
            )

    snapshot_log_probs = torch.log_softmax(
        rollout.policy_logits_snapshot, dim=-1
    )[rollout.states, rollout.actions]
    snapshot_error = float(
        torch.max(torch.abs(snapshot_log_probs - rollout.old_log_probabilities))
    )
    if not torch.equal(rollout.old_log_probabilities, old_log_probs_before):
        raise AssertionError("stored behavior-policy log probabilities were mutated")
    if snapshot_error != 0:
        raise AssertionError("stored old log-probs do not match the policy snapshot")
    return {
        "optimizer_steps": optimizer_steps,
        "advantage_mean_before_normalization": float(advantage_mean),
        "advantage_std_before_normalization": float(advantage_std),
        "return_mean": float(flat_returns.mean()),
        "initial_minibatch_losses": loss_rows[0],
        "final_minibatch_losses": loss_rows[-1],
        "stored_old_log_probabilities_unchanged": True,
        "snapshot_log_probability_max_error": snapshot_error,
        "post_update_objective": _evaluate_rollout_objective(
            model,
            rollout,
            normalized_advantages,
            clip_epsilon=clip_epsilon,
        ),
    }


def run_smoke(
    *,
    iterations: int = 6,
    episodes_per_iteration: int = 128,
    epochs: int = 4,
    minibatch_size: int = 64,
    learning_rate: float = 0.05,
) -> dict[str, Any]:
    """Execute reproducible categorical rollout and PPO optimizer iterations."""

    iteration_count = _positive_integer(iterations, "iterations")
    episodes = _positive_integer(episodes_per_iteration, "episodes_per_iteration")
    epoch_count = _positive_integer(epochs, "epochs")
    batch_size = _positive_integer(minibatch_size, "minibatch_size")
    rate = _finite_positive(learning_rate, "learning_rate")
    torch.manual_seed(47)
    rollout_generator = torch.Generator(device="cpu").manual_seed(48)
    optimizer_generator = torch.Generator(device="cpu").manual_seed(49)
    model = TinyActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    initial_policy = torch.softmax(model.policy_logits.detach(), dim=-1).tolist()
    initial_values = model.state_values.detach().tolist()
    initial_expected_return = exact_expected_return(model)
    initial_policy_parameters = model.policy_logits.detach().clone()
    initial_value_parameters = model.state_values.detach().clone()

    iteration_reports: list[dict[str, Any]] = []
    rollout_reward_means: list[float] = []
    for iteration in range(iteration_count):
        rollout = collect_rollout(
            model, episodes=episodes, generator=rollout_generator
        )
        rollout_reward_means.append(float(rollout.rewards.sum(dim=1).mean()))
        update = optimize_rollout(
            model,
            rollout,
            optimizer,
            epochs=epoch_count,
            minibatch_size=batch_size,
            clip_epsilon=0.2,
            gamma=0.95,
            gae_lambda=0.95,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            generator=optimizer_generator,
        )
        iteration_reports.append(
            {
                "iteration": iteration,
                "rollout_episode_count": episodes,
                "rollout_action_count": int(rollout.valid_mask.sum()),
                "terminated_transition_count": int(rollout.terminated.sum()),
                "truncated_transition_count": int(rollout.truncated.sum()),
                "old_log_probabilities_require_grad": (
                    rollout.old_log_probabilities.requires_grad
                ),
                "rollout_reward_mean": rollout_reward_means[-1],
                **update,
            }
        )

    final_policy = torch.softmax(model.policy_logits.detach(), dim=-1).tolist()
    final_values = model.state_values.detach().tolist()
    final_expected_return = exact_expected_return(model)
    all_old_log_probs_unchanged = all(
        report["stored_old_log_probabilities_unchanged"]
        for report in iteration_reports
    )
    all_snapshot_errors_zero = all(
        report["snapshot_log_probability_max_error"] == 0
        for report in iteration_reports
    )
    return {
        "environment": {
            "state_count": 2,
            "action_count": 2,
            "horizon": 2,
            "reward_rule": "state 0 rewards action 1; state 1 rewards action 0",
            "maximum_undiscounted_return": 2.0,
        },
        "configuration": {
            "seed": 47,
            "dtype": "float64",
            "iterations": iteration_count,
            "episodes_per_iteration": episodes,
            "epochs_per_rollout": epoch_count,
            "minibatch_size": batch_size,
            "learning_rate": rate,
            "gamma": 0.95,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.01,
        },
        "initial_exact_expected_return": initial_expected_return,
        "final_exact_expected_return": final_expected_return,
        "initial_policy_probabilities": initial_policy,
        "final_policy_probabilities": final_policy,
        "initial_state_values": initial_values,
        "final_state_values": final_values,
        "first_rollout_reward_mean": rollout_reward_means[0],
        "last_rollout_reward_mean": rollout_reward_means[-1],
        "total_optimizer_steps": sum(
            int(report["optimizer_steps"]) for report in iteration_reports
        ),
        "policy_parameters_changed": not torch.equal(
            model.policy_logits.detach(), initial_policy_parameters
        ),
        "value_parameters_changed": not torch.equal(
            model.state_values.detach(), initial_value_parameters
        ),
        "all_stored_old_log_probabilities_unchanged": all_old_log_probs_unchanged,
        "all_snapshot_log_probability_errors_zero": all_snapshot_errors_zero,
        "iterations": iteration_reports,
        "scope": {
            "device": "CPU",
            "authored_two_state_environment": True,
            "on_policy_categorical_sampling_executed": True,
            "torch_policy_and_value_forward_executed": True,
            "gae_and_minibatch_optimizer_executed": True,
            "time_limit_truncation_executed": False,
            "language_model_or_tokenizer_executed": False,
            "reward_model_executed": False,
            "reference_policy_kl_controller_executed": False,
            "gpu_or_distributed_execution": False,
            "target_model_quality_or_safety_proved": False,
            "production_ppo_stability_proved": False,
        },
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
