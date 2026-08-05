"""Small, testable helpers around Hugging Face Transformers contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def parameter_report(model: Any) -> dict[str, int | float]:
    """Count total/trainable parameters and their storage bytes."""
    parameters = list(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    storage_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "parameter_storage_bytes": storage_bytes,
    }


def render_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render with the checkpoint's template and fail loudly if unavailable."""
    if not messages:
        raise ValueError("messages cannot be empty")
    for index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise ValueError(f"message {index} must contain exactly role and content")
        if not message["role"] or not message["content"]:
            raise ValueError(f"message {index} role and content cannot be empty")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer has no chat_template; do not guess a model-specific format")
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("chat template returned an empty or non-string value")
    return rendered
