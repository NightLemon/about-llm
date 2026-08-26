# ruff: noqa: RUF001 -- Full-width punctuation is part of the Chinese sample.

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "trace_qwen3_tokenizer.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qwen3_tokenizer_trace", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeQwen3Tokenizer:
    chat_template = "qwen3 teaching template"
    all_special_ids: ClassVar[list[int]] = [10, 11]
    special_tokens_map: ClassVar[dict[str, str]] = {
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
    }
    rendered = (
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    token_ids: ClassVar[list[int]] = [10, 20, 11, 10, 21]
    tokens: ClassVar[list[str]] = [
        "<|im_start|>",
        "user\\n你好",
        "<|im_end|>",
        "<|im_start|>",
        "assistant",
    ]
    added_vocab: ClassVar[dict[str, int]] = {
        "<|im_start|>": 10,
        "<|im_end|>": 11,
        "assistant": 21,
    }

    def __init__(self) -> None:
        self.template_calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | list[int]:
        assert messages == [
            {
                "role": "user",
                "content": "请用一句话解释：为什么生成下一个 token 时可以复用 KV Cache？",
            }
        ]
        self.template_calls.append(
            {
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        return list(self.token_ids) if tokenize else self.rendered

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert text == self.rendered
        assert add_special_tokens is False
        return {"input_ids": list(self.token_ids)}

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        assert token_ids == self.token_ids
        return list(self.tokens)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self.added_vocab)

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        if token_ids == self.token_ids:
            return self.rendered
        pieces = {
            10: "<|im_start|>",
            20: "user\n你好",
            11: "<|im_end|>",
            21: "assistant",
        }
        return "".join(pieces[token_id] for token_id in token_ids)


def test_trace_connects_message_template_ids_and_round_trip() -> None:
    module = _load_script()
    tokenizer = FakeQwen3Tokenizer()

    report = module.trace_chat_message(tokenizer)

    assert report["schema"] == "about-llm.qwen3-tokenizer-trace.v1"
    assert report["model"]["target_repository"] == "Qwen/Qwen3-0.6B"
    assert report["model"]["target_revision"] == (
        "c1899de289a04d12100db370d81485cdf75e47ca"
    )
    assert isinstance(report["runtime"]["python"], str)
    assert isinstance(report["runtime"]["transformers"], str)
    assert report["rendered_prompt"] == tokenizer.rendered
    assert report["tokenization"]["token_ids"] == tokenizer.token_ids
    assert report["tokenization"]["token_count"] == 5
    assert report["tokenization"]["template_ids_match_rendered_encoding"] is True
    assert [
        row["tokenizer_kind"] for row in report["tokenization"]["tokens"]
    ] == [
        "special",
        "regular",
        "special",
        "special",
        "added",
    ]
    assert report["tokenization"]["tokens"][1]["decoded_piece"] == "user\n你好"
    assert report["tokenizer_metadata"]["added_token_count"] == 3
    assert report["round_trip"]["matches_rendered_prompt"] is True
    assert report["scope"] == {
        "qwen3_tokenizer_and_chat_template_executed": True,
        "model_weights_loaded": False,
        "nano_vllm_executed": False,
        "gpu_required": False,
    }
    assert tokenizer.template_calls == [
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
    ]


def test_human_output_explains_what_ran() -> None:
    module = _load_script()
    report = module.trace_chat_message(FakeQwen3Tokenizer())

    output = module.format_human_readable(report)

    assert "Qwen3 对话分词追踪" in output
    assert "Qwen/Qwen3-0.6B@c1899de" in output
    assert "Token IDs（共 5 个）" in output
    assert "模板直接生成的 IDs 与渲染后再次编码：一致" in output
    assert "没有加载模型权重" in output


def test_local_snapshot_is_loaded_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> FakeQwen3Tokenizer:
            calls.append((source, kwargs))
            return FakeQwen3Tokenizer()

    monkeypatch.setattr(module, "AutoTokenizer", FakeAutoTokenizer)
    tokenizer, source, source_kind = module.load_tokenizer(
        model_snapshot=tmp_path,
        local_files_only=False,
    )

    assert isinstance(tokenizer, FakeQwen3Tokenizer)
    assert source == str(tmp_path.resolve())
    assert source_kind == "local_snapshot"
    assert calls == [
        (
            str(tmp_path.resolve()),
            {"local_files_only": True, "trust_remote_code": False},
        )
    ]


def test_repository_load_uses_the_full_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> FakeQwen3Tokenizer:
            calls.append((source, kwargs))
            return FakeQwen3Tokenizer()

    monkeypatch.setattr(module, "AutoTokenizer", FakeAutoTokenizer)
    module.load_tokenizer(model_snapshot=None, local_files_only=True)

    assert calls == [
        (
            "Qwen/Qwen3-0.6B",
            {
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]


def test_trace_rejects_non_integer_template_output() -> None:
    module = _load_script()

    class BrokenTokenizer(FakeQwen3Tokenizer):
        def apply_chat_template(
            self,
            messages: Sequence[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            enable_thinking: bool,
        ) -> str | list[str]:
            if tokenize:
                return ["not-an-id"]
            return self.rendered

    with pytest.raises(RuntimeError, match="integer token IDs"):
        module.trace_chat_message(BrokenTokenizer())
