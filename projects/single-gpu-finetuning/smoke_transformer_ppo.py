"""用微型 Transformer actor-critic 跑 token-level PPO，并穷举精确 task reward。

模型从 BOS 开始生成最多两个 token，目标是产生指定 token。实验采集自回归 rollout、处理 EOS
与 padding、执行 clipped PPO，再枚举全部短序列检查期望任务回报是否改善。
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import Tensor

from about_llm.finetuning.ppo_transformer import (
    TinyCausalActorCritic,
    TransformerRollout,
    finite_non_negative,
    freeze_copy,
    optimize_token_rollout,
    positive_integer,
)

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
TARGET_TOKEN_ID = 4
VOCAB_SIZE = 6
HORIZON = 2
MAX_CONTEXT_LENGTH = 2


def _step_contexts(previous_actions: Tensor | None, episodes: int) -> tuple[Tensor, Tensor]:
    """为当前生成步构造固定长度 context 与 attention mask。"""

    contexts = torch.full(
        (episodes, MAX_CONTEXT_LENGTH), PAD_TOKEN_ID, dtype=torch.long
    )
    attention_masks = torch.zeros_like(contexts)
    contexts[:, 0] = BOS_TOKEN_ID
    attention_masks[:, 0] = 1
    if previous_actions is not None:
        contexts[:, 1] = previous_actions
        attention_masks[:, 1] = 1
    return contexts, attention_masks


@torch.inference_mode()
def exact_expected_task_reward(model: TinyCausalActorCritic) -> float:
    """枚举所有两步 token 路径，精确计算目标 token 奖励期望。"""

    """Enumerate all first-token branches for the two-action reward horizon."""

    model.eval()
    first_context, first_mask = _step_contexts(None, 1)
    first_logits, _ = model.forward_contexts(first_context, first_mask)
    first_probabilities = torch.softmax(first_logits[0], dim=-1)
    first_actions = torch.arange(VOCAB_SIZE, dtype=torch.long)
    second_contexts, second_masks = _step_contexts(first_actions, VOCAB_SIZE)
    second_logits, _ = model.forward_contexts(second_contexts, second_masks)
    second_target_probabilities = torch.softmax(second_logits, dim=-1)[
        :, TARGET_TOKEN_ID
    ]
    expected_second_reward = torch.sum(
        first_probabilities * second_target_probabilities
    )
    return float(first_probabilities[TARGET_TOKEN_ID] + expected_second_reward)


@torch.inference_mode()
def collect_token_rollout(
    model: TinyCausalActorCritic,
    reference: TinyCausalActorCritic,
    *,
    episodes: int,
    kl_coefficient: float,
    generator: torch.Generator,
) -> TransformerRollout:
    """逐 token 采样并保存 old log-prob、value、mask 与终止状态。"""

    """Sample two tokens and bind every action to behavior/reference statistics."""

    episode_count = positive_integer(episodes, "episodes")
    coefficient = finite_non_negative(kl_coefficient, "kl_coefficient")
    behavior = freeze_copy(model)
    contexts = torch.zeros(
        (episode_count, HORIZON, MAX_CONTEXT_LENGTH), dtype=torch.long
    )
    attention_masks = torch.zeros_like(contexts)
    actions = torch.zeros((episode_count, HORIZON), dtype=torch.long)
    old_log_probabilities = torch.zeros(
        (episode_count, HORIZON), dtype=torch.float32
    )
    reference_log_probabilities = torch.zeros_like(old_log_probabilities)
    old_values = torch.zeros_like(old_log_probabilities)
    exact_categorical_kls = torch.zeros_like(old_log_probabilities)

    previous_actions: Tensor | None = None
    for step in range(HORIZON):
        step_contexts, step_masks = _step_contexts(previous_actions, episode_count)
        contexts[:, step] = step_contexts
        attention_masks[:, step] = step_masks
        behavior_logits, behavior_values = behavior.forward_contexts(
            step_contexts, step_masks
        )
        reference_logits, _ = reference.forward_contexts(step_contexts, step_masks)
        behavior_log_probs = torch.log_softmax(behavior_logits, dim=-1)
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
        behavior_probabilities = torch.exp(behavior_log_probs)
        step_actions = torch.multinomial(
            behavior_probabilities, 1, replacement=True, generator=generator
        ).squeeze(1)
        actions[:, step] = step_actions
        old_log_probabilities[:, step] = behavior_log_probs.gather(
            1, step_actions.unsqueeze(1)
        ).squeeze(1)
        reference_log_probabilities[:, step] = reference_log_probs.gather(
            1, step_actions.unsqueeze(1)
        ).squeeze(1)
        old_values[:, step] = behavior_values
        exact_categorical_kls[:, step] = torch.sum(
            behavior_probabilities * (behavior_log_probs - reference_log_probs),
            dim=-1,
        )
        previous_actions = step_actions

    task_rewards = (actions == TARGET_TOKEN_ID).to(torch.float32)
    sampled_reference_log_ratios = (
        old_log_probabilities - reference_log_probabilities
    )
    shaped_rewards = task_rewards - coefficient * sampled_reference_log_ratios
    next_values = torch.zeros_like(old_values)
    next_values[:, :-1] = old_values[:, 1:]
    valid_mask = torch.ones_like(actions, dtype=torch.bool)
    terminated = torch.zeros_like(actions, dtype=torch.bool)
    terminated[:, -1] = True
    truncated = torch.zeros_like(actions, dtype=torch.bool)
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
    )


def run_smoke(
    *,
    iterations: int = 6,
    episodes_per_iteration: int = 64,
    epochs: int = 3,
    minibatch_size: int = 64,
    learning_rate: float = 0.01,
    kl_coefficient: float = 0.02,
) -> dict[str, Any]:
    """采集 rollout、执行 PPO 更新，并比较精确期望任务回报。"""

    """Execute reproducible Transformer token rollout and PPO updates on CPU."""

    iteration_count = positive_integer(iterations, "iterations")
    episodes = positive_integer(episodes_per_iteration, "episodes_per_iteration")
    epoch_count = positive_integer(epochs, "epochs")
    batch_size = positive_integer(minibatch_size, "minibatch_size")
    rate = finite_non_negative(learning_rate, "learning_rate")
    if rate == 0:
        raise ValueError("learning_rate must be positive")
    coefficient = finite_non_negative(kl_coefficient, "kl_coefficient")
    torch.manual_seed(61)
    # rollout 与 minibatch shuffle 使用独立 RNG，便于分别复现采样和优化顺序。
    rollout_generator = torch.Generator(device="cpu").manual_seed(62)
    optimizer_generator = torch.Generator(device="cpu").manual_seed(63)
    model = TinyCausalActorCritic(
        vocab_size=VOCAB_SIZE,
        max_positions=4,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=2,
        pad_token_id=PAD_TOKEN_ID,
    ).to("cpu")
    reference = freeze_copy(model)
    reference_before = {
        name: parameter.detach().clone()
        for name, parameter in reference.named_parameters()
    }
    initial_model_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    initial_exact_reward = exact_expected_task_reward(model)
    iteration_reports: list[dict[str, Any]] = []
    for iteration in range(iteration_count):
        rollout = collect_token_rollout(
            model,
            reference,
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
            bootstrap_truncated=True,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            generator=optimizer_generator,
        )
        iteration_reports.append(
            {
                "iteration": iteration,
                "rollout_episode_count": episodes,
                "rollout_action_count": int(rollout.valid_mask.sum()),
                "sampled_task_reward_mean": float(
                    rollout.task_rewards.sum(dim=1).mean()
                ),
                "sampled_reference_log_ratio_mean": float(
                    rollout.sampled_reference_log_ratios.mean()
                ),
                "exact_categorical_kl_at_sampled_states_mean": float(
                    rollout.exact_categorical_kls.mean()
                ),
                "terminated_transition_count": int(rollout.terminated.sum()),
                "truncated_transition_count": int(rollout.truncated.sum()),
                "old_log_probabilities_require_grad": (
                    rollout.old_log_probabilities.requires_grad
                ),
                **update,
            }
        )

    final_exact_reward = exact_expected_task_reward(model)
    for name, parameter in reference.named_parameters():
        if not torch.equal(parameter.detach(), reference_before[name]):
            raise AssertionError(f"reference parameter changed: {name}")
    changed_parameter_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), initial_model_parameters[name])
    )
    backbone_parameters_changed = any(
        name.startswith("backbone.") for name in changed_parameter_names
    )
    policy_head_parameters_changed = any(
        name.startswith("policy_head.") for name in changed_parameter_names
    )
    value_head_parameters_changed = any(
        name.startswith("value_head.") for name in changed_parameter_names
    )
    return {
        "model_class": "GPT2Model+policy_head+value_head",
        "configuration": {
            "seed": 61,
            "vocab_size": VOCAB_SIZE,
            "bos_token_id": BOS_TOKEN_ID,
            "target_token_id": TARGET_TOKEN_ID,
            "horizon": HORIZON,
            "iterations": iteration_count,
            "episodes_per_iteration": episodes,
            "epochs_per_rollout": epoch_count,
            "minibatch_size": batch_size,
            "learning_rate": rate,
            "kl_coefficient": coefficient,
            "gamma": 0.95,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
        },
        "initial_exact_expected_task_reward": initial_exact_reward,
        "final_exact_expected_task_reward": final_exact_reward,
        "first_sampled_task_reward_mean": iteration_reports[0][
            "sampled_task_reward_mean"
        ],
        "last_sampled_task_reward_mean": iteration_reports[-1][
            "sampled_task_reward_mean"
        ],
        "total_optimizer_steps": sum(
            int(report["optimizer_steps"]) for report in iteration_reports
        ),
        "reference_parameters_unchanged": True,
        "backbone_parameters_changed": backbone_parameters_changed,
        "policy_head_parameters_changed": policy_head_parameters_changed,
        "value_head_parameters_changed": value_head_parameters_changed,
        "all_stored_old_log_probabilities_unchanged": all(
            report["stored_old_log_probabilities_unchanged"]
            for report in iteration_reports
        ),
        "maximum_snapshot_log_probability_error": max(
            float(report["snapshot_log_probability_max_error"])
            for report in iteration_reports
        ),
        "iterations": iteration_reports,
        "scope": {
            "device": "CPU",
            "integer_token_ids_without_tokenizer": True,
            "random_tiny_gpt2_backbone_executed": True,
            "autoregressive_token_sampling_executed": True,
            "frozen_reference_forward_executed": True,
            "sampled_reference_log_ratio_reward_executed": True,
            "exact_two_step_task_reward_enumerated": True,
            "gae_and_transformer_optimizer_executed": True,
            "learned_reward_model_executed": False,
            "natural_language_quality_proved": False,
            "time_limit_truncation_executed": False,
            "checkpoint_or_resume_executed": False,
            "cuda_or_distributed_execution": False,
            "target_llm_ppo_quality_or_safety_proved": False,
        },
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
