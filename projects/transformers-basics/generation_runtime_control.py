"""离线观察 Transformers ``generate`` 的 EOS 与长度停止规则。

随机模型本身无法稳定生成指定 token，因此实验插入一个 LogitsProcessor，按预定计划强制
每一步的输出。这样可以把注意力放在三个问题上：EOS 列表如何生效、调用参数如何覆盖
GenerationConfig，以及没有 EOS 时 ``max_new_tokens`` 如何停止生成。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch
import transformers
from transformers import (
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
    LogitsProcessor,
    LogitsProcessorList,
)

from about_llm.integrations.transformers_tools import parameter_report


@dataclass
class ForcedTokenPlan(LogitsProcessor):
    """把每一步的全部候选分数替换为预定的唯一 token，并记录调用轨迹。

    这个处理器只控制“下一 token 是谁”，停止条件仍由 Transformers 的真实生成循环判断。
    因此我们可以在不依赖随机模型输出的情况下单独验证停止语义。
    """

    prompt_length: int
    token_plan: tuple[int, ...]
    vocabulary_size: int
    trace: list[dict[str, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """在生成开始前检查 prompt、计划和词表范围。"""

        if isinstance(self.prompt_length, bool) or self.prompt_length <= 0:
            raise ValueError("prompt_length must be a positive integer")
        if not self.token_plan:
            raise ValueError("token_plan cannot be empty")
        if isinstance(self.vocabulary_size, bool) or self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be a positive integer")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < self.vocabulary_size
            for token_id in self.token_plan
        ):
            raise ValueError("token_plan ids must be integers inside the vocabulary")

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """根据当前序列长度选择计划 token，并覆盖本步 logits。"""

        # input_ids 已包含 prompt 和此前生成的 token，两者长度差就是当前生成步编号。
        step = int(input_ids.shape[1]) - self.prompt_length
        if not 0 <= step < len(self.token_plan):
            raise RuntimeError(
                "Transformers requested a token beyond the authored plan; expected "
                "EOS or length stopping to have occurred"
            )
        if scores.ndim != 2 or scores.shape[1] != self.vocabulary_size:
            raise RuntimeError("unexpected next-token score shape")
        token_id = self.token_plan[step]
        # 其余 token 设为 -inf、目标 token 设为 0，贪心选择必然得到目标 token。
        forced = torch.full_like(scores, float("-inf"))
        forced[:, token_id] = 0
        self.trace.append(
            {
                "generation_step": step,
                "input_length_before_step": int(input_ids.shape[1]),
                "forced_token_id": token_id,
            }
        )
        return forced


def _run_case(
    model: GPT2LMHeadModel,
    *,
    prompt: torch.Tensor,
    generation_config: GenerationConfig,
    token_plan: tuple[int, ...],
    call_eos_token_id: int | None = None,
    call_max_new_tokens: int | None = None,
) -> dict[str, object]:
    """运行一组停止条件，并拆分 prompt 与本次新生成的 token。"""

    # 每个 case 新建 processor，轨迹不会在多个对照之间串联。
    processor = ForcedTokenPlan(
        prompt_length=int(prompt.shape[1]),
        token_plan=token_plan,
        vocabulary_size=int(model.config.vocab_size),
    )
    # 只有显式传入的调用级参数才放进 overrides，用来观察它们是否覆盖配置对象。
    overrides: dict[str, Any] = {}
    if call_eos_token_id is not None:
        overrides["eos_token_id"] = call_eos_token_id
    if call_max_new_tokens is not None:
        overrides["max_new_tokens"] = call_max_new_tokens
    # 这里运行 Transformers GenerationMixin 的真实循环，只替换 token 选择分数。
    with torch.inference_mode():
        output = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            generation_config=generation_config,
            logits_processor=LogitsProcessorList([processor]),
            return_dict_in_generate=True,
            **overrides,
        )
    sequences = getattr(output, "sequences", None)
    if not isinstance(sequences, torch.Tensor):
        raise RuntimeError("generate did not return tensor sequences")
    sequences = sequences.detach().cpu()
    if sequences.ndim != 2 or sequences.shape[0] != 1:
        raise RuntimeError("control expects exactly one generated sequence")
    prompt_ids = prompt[0].detach().cpu().tolist()
    sequence_ids = sequences[0].tolist()
    if sequence_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("generate changed the prompt prefix")
    # generate 返回 prompt 与续写的拼接序列，因此按原 prompt 长度切出新 token。
    generated_ids = sequence_ids[len(prompt_ids) :]
    return {
        "prompt_token_ids": prompt_ids,
        "generation_config_eos_token_ids": list(generation_config.eos_token_id),
        "generation_config_max_new_tokens": generation_config.max_new_tokens,
        "call_overrides": overrides,
        "authored_forced_token_plan": list(token_plan),
        "generated_token_ids": generated_ids,
        "new_token_count": len(generated_ids),
        "processor_trace": processor.trace,
        "finish_reason_source": (
            "inferred from the controlled token plan and configured EOS/length; "
            "Transformers generate output does not expose a provider-style finish_reason"
        ),
    }


def run_control() -> dict[str, object]:
    """依次运行 EOS 集合、调用级覆盖和长度上限三组对照。"""

    # 模型只负责让 generate 走过真实前向路径；随机种子固定其参数身份。
    torch.manual_seed(71)
    model_config = GPT2Config(  # type: ignore[no-untyped-call]
        vocab_size=16,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )
    model = GPT2LMHeadModel(model_config)  # type: ignore[no-untyped-call]
    model.eval()  # type: ignore[no-untyped-call]
    prompt = torch.tensor([[1, 7]], dtype=torch.long)
    generation_config = GenerationConfig(  # type: ignore[no-untyped-call]
        bos_token_id=1,
        eos_token_id=[2, 3],
        pad_token_id=0,
        do_sample=False,
        max_new_tokens=5,
        use_cache=False,
    )
    # 保存副本，实验结束后还要检查 generate 没有原地修改调用者持有的配置。
    generation_config_before = generation_config.to_dict()

    # token 3 属于配置中的 EOS 列表，生成 [4, 3] 后应立即停止。
    eos_set_case = _run_case(
        model,
        prompt=prompt,
        generation_config=generation_config,
        token_plan=(4, 3),
    )
    # 调用级 eos_token_id=5 覆盖配置中的 [2, 3]，所以 token 3 不再触发停止。
    call_override_case = _run_case(
        model,
        prompt=prompt,
        generation_config=generation_config,
        token_plan=(3, 5),
        call_eos_token_id=5,
        call_max_new_tokens=4,
    )
    # 计划中没有 EOS；调用级 max_new_tokens=2 成为唯一停止原因。
    length_cap_case = _run_case(
        model,
        prompt=prompt,
        generation_config=generation_config,
        token_plan=(4, 6),
        call_max_new_tokens=2,
    )
    # 用显式断言把“观察到的输出”变成实验必须满足的语义，不只是在报告里展示。
    if eos_set_case["generated_token_ids"] != [4, 3]:
        raise RuntimeError("generation-config EOS set did not stop on token 3")
    if call_override_case["generated_token_ids"] != [3, 5]:
        raise RuntimeError("call-level EOS override did not replace the config EOS set")
    if length_cap_case["generated_token_ids"] != [4, 6]:
        raise RuntimeError("call-level max_new_tokens did not stop at two tokens")
    if generation_config.to_dict() != generation_config_before:
        raise RuntimeError("generate mutated the caller-owned GenerationConfig")

    return {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "parameter_report": parameter_report(model),
        "model_config": {
            "vocab_size": model_config.vocab_size,
            "n_positions": model_config.n_positions,
            "n_embd": model_config.n_embd,
            "n_layer": model_config.n_layer,
            "n_head": model_config.n_head,
        },
        "cases": {
            "generation_config_eos_set": eos_set_case,
            "call_level_eos_override": call_override_case,
            "call_level_length_cap": length_cap_case,
        },
        "assertions": {
            "generation_config_eos_list_stopped_on_member_3": True,
            "call_eos_5_overrode_config_eos_set_2_3": True,
            "config_eos_3_did_not_stop_override_case": True,
            "call_max_new_tokens_2_stopped_without_eos": True,
            "caller_generation_config_not_mutated": True,
        },
        "scope": {
            "real_transformers_generation_mixin_executed": True,
            "real_tiny_gpt2_forward_executed": True,
            "authored_logits_processor_overrode_all_next_token_scores": True,
            "random_untrained_model_used": True,
            "real_tokenizer_or_chat_template_executed": False,
            "public_checkpoint_or_remote_code_loaded": False,
            "vllm_or_provider_runtime_executed": False,
            "model_quality_latency_throughput_or_gpu_behavior_proved": False,
            "provider_style_finish_reason_observed": False,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_control(), ensure_ascii=False, indent=2))
