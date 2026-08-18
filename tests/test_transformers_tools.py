from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from about_llm.integrations.transformers_tools import parameter_report, render_chat

pytestmark = pytest.mark.contract


class FakeParameter:
    def __init__(self, elements: int, bytes_per_element: int, trainable: bool) -> None:
        self.elements = elements
        self.bytes_per_element = bytes_per_element
        self.requires_grad = trainable

    def numel(self) -> int:
        return self.elements

    def element_size(self) -> int:
        return self.bytes_per_element


class FakeModel:
    def parameters(self) -> list[FakeParameter]:
        return [FakeParameter(10, 2, True), FakeParameter(30, 1, False)]


class FakeTokenizer:
    chat_template = "available"

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        suffix = "<assistant>" if add_generation_prompt else ""
        return "".join(f"<{item['role']}>{item['content']}" for item in messages) + suffix


def test_parameter_report_counts_storage_not_runtime_memory() -> None:
    report = parameter_report(FakeModel())
    assert report == {
        "total_parameters": 40,
        "trainable_parameters": 10,
        "trainable_fraction": 0.25,
        "parameter_storage_bytes": 50,
    }


def test_render_chat_uses_template_and_validates_messages() -> None:
    rendered = render_chat(FakeTokenizer(), [{"role": "user", "content": "hello"}])
    assert rendered == "<user>hello<assistant>"

    with pytest.raises(ValueError, match="exactly role and content"):
        render_chat(FakeTokenizer(), [{"role": "user", "content": "x", "extra": "y"}])


def test_render_chat_rejects_missing_template() -> None:
    tokenizer: Any = FakeTokenizer()
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="no chat_template"):
        render_chat(tokenizer, [{"role": "user", "content": "hello"}])
