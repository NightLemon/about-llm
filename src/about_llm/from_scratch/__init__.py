"""Transparent reference implementations of core LLM components."""

from typing import TYPE_CHECKING, cast

from about_llm.from_scratch.attention_numpy import (
    OnlineAttentionResult,
    apply_rope,
    blockwise_online_attention,
    causal_mask,
    grouped_query_attention,
    rms_norm,
    scaled_dot_product_attention,
)
from about_llm.from_scratch.moe_routing import (
    MoERoutingResult,
    route_topk_capacity,
    routed_linear_expert_forward,
)
from about_llm.from_scratch.tokenizer import ByteBPETokenizer, ByteTokenizer

if TYPE_CHECKING:
    from about_llm.from_scratch.gpt_torch import GPTConfig, MiniGPT
    from about_llm.from_scratch.moe_training import (
        TrainableMoEForward,
        TrainableTopKMoE,
    )


def __getattr__(name: str) -> object:
    if name in {"GPTConfig", "MiniGPT"}:
        from about_llm.from_scratch import gpt_torch

        return cast(object, getattr(gpt_torch, name))
    if name in {"TrainableMoEForward", "TrainableTopKMoE"}:
        from about_llm.from_scratch import moe_training

        return cast(object, getattr(moe_training, name))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ByteBPETokenizer",
    "ByteTokenizer",
    "GPTConfig",
    "MiniGPT",
    "MoERoutingResult",
    "OnlineAttentionResult",
    "TrainableMoEForward",
    "TrainableTopKMoE",
    "apply_rope",
    "blockwise_online_attention",
    "causal_mask",
    "grouped_query_attention",
    "rms_norm",
    "route_topk_capacity",
    "routed_linear_expert_forward",
    "scaled_dot_product_attention",
]
