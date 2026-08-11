"""Deterministic offline control for Transformers generation stopping semantics."""

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
    """Replace every next-token score with one authored deterministic choice."""

    prompt_length: int
    token_plan: tuple[int, ...]
    vocabulary_size: int
    trace: list[dict[str, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
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
        step = int(input_ids.shape[1]) - self.prompt_length
        if not 0 <= step < len(self.token_plan):
            raise RuntimeError(
                "Transformers requested a token beyond the authored plan; expected "
                "EOS or length stopping to have occurred"
            )
        if scores.ndim != 2 or scores.shape[1] != self.vocabulary_size:
            raise RuntimeError("unexpected next-token score shape")
        token_id = self.token_plan[step]
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
    processor = ForcedTokenPlan(
        prompt_length=int(prompt.shape[1]),
        token_plan=token_plan,
        vocabulary_size=int(model.config.vocab_size),
    )
    overrides: dict[str, Any] = {}
    if call_eos_token_id is not None:
        overrides["eos_token_id"] = call_eos_token_id
    if call_max_new_tokens is not None:
        overrides["max_new_tokens"] = call_max_new_tokens
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
    """Execute EOS-set, call-override, and length-cap paths on a random tiny model."""

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
    generation_config_before = generation_config.to_dict()

    eos_set_case = _run_case(
        model,
        prompt=prompt,
        generation_config=generation_config,
        token_plan=(4, 3),
    )
    call_override_case = _run_case(
        model,
        prompt=prompt,
        generation_config=generation_config,
        token_plan=(3, 5),
        call_eos_token_id=5,
        call_max_new_tokens=4,
    )
    length_cap_case = _run_case(
        model,
        prompt=prompt,
        generation_config=generation_config,
        token_plan=(4, 6),
        call_max_new_tokens=2,
    )
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
