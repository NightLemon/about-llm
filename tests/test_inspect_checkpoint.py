from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "inspect_checkpoint.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inspect_checkpoint_project", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeConfig:
    model_type = "authored-test-decoder"
    architectures: ClassVar[list[str]] = ["AuthoredTestForCausalLM"]
    vocab_size = 32000
    max_position_embeddings = 8192
    _commit_hash = "abc123"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architectures": self.architectures,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }


class FakeAutoConfig:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeConfig:
        cls.calls.append((model_id, kwargs))
        return FakeConfig()


class BaseTokenizer:
    chat_template = None
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    decoder_start_token_id = None

    def __len__(self) -> int:
        return 32000


class InstructTokenizer:
    chat_template = "authored template"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    decoder_start_token_id = None

    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    def __len__(self) -> int:
        return 32000

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[int]:
        assert messages == [{"role": "user", "content": "用一句话解释 attention。"}]
        self.calls.append((tokenize, add_generation_prompt))
        if tokenize:
            return [101, 202, 303]
        return "<user>attention</user><assistant>"


class FakeAutoTokenizer:
    tokenizer: ClassVar[Any] = BaseTokenizer()
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    @classmethod
    def from_pretrained(
        cls, model_id: str, **kwargs: Any
    ) -> Any:
        cls.calls.append((model_id, kwargs))
        return cls.tokenizer


class FakeGenerationConfig:
    _commit_hash = "generation-commit-456"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bos_token_id": 1,
            "eos_token_id": [2, 3],
            "pad_token_id": 0,
            "do_sample": False,
            "max_new_tokens": 64,
        }


class FakeGenerationConfigLoader:
    available: ClassVar[bool] = True
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeGenerationConfig:
        cls.calls.append((model_id, kwargs))
        if not cls.available:
            raise OSError("authored missing-generation-config fixture")
        return FakeGenerationConfig()


def _install_fakes(
    module: ModuleType, tokenizer: Any, *, generation_config_available: bool = True
) -> None:
    FakeAutoConfig.calls = []
    FakeAutoTokenizer.calls = []
    FakeAutoTokenizer.tokenizer = tokenizer
    FakeGenerationConfigLoader.calls = []
    FakeGenerationConfigLoader.available = generation_config_available
    dynamic_module: Any = module
    dynamic_module.AutoConfig = FakeAutoConfig
    dynamic_module.AutoTokenizer = FakeAutoTokenizer
    dynamic_module.GenerationConfig = FakeGenerationConfigLoader


def test_base_tokenizer_without_chat_template_still_inspects_config() -> None:
    module = _load_script()
    _install_fakes(module, BaseTokenizer(), generation_config_available=False)

    report = module.inspect("org/base-model", "pinned-revision")

    assert report["chat_template_available"] is False
    assert report["rendered_prompt"] is None
    assert report["rendered_token_ids"] is None
    assert report["token_count"] is None
    assert report["resolved_config_commit"] == "abc123"
    assert report["normalized_config_snapshot_source"] == (
        "AutoConfig.to_dict() after Transformers loading; may include library defaults "
        "and runtime metadata, so its fingerprint is not a hash of raw config.json bytes"
    )
    assert report["generation_config_status"] == "unavailable_or_load_error"
    assert report["generation_config_error_type"] == "OSError"
    assert report["resolved_generation_config_commit"] is None
    assert report["generation_protocol_contract"]["observations"] == [
        "generation_config_snapshot_unavailable"
    ]
    assert report["generation_protocol_contract"][
        "effective_runtime_contract_proved"
    ] is False
    assert report["normalized_config_contract"]["standard_kv_layout"] == {
        "applicable": True,
        "reason": (
            "explicit fields satisfy the ideal standard MHA/GQA/MQA layout contract"
        ),
        "attention_kind": "gqa",
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "query_heads_per_kv_head": 4,
    }
    assert FakeAutoConfig.calls == [
        (
            "org/base-model",
            {"revision": "pinned-revision", "trust_remote_code": False},
        )
    ]
    assert FakeAutoTokenizer.calls == FakeAutoConfig.calls
    assert FakeGenerationConfigLoader.calls == [
        ("org/base-model", {"revision": "pinned-revision"})
    ]


def test_instruct_tokenizer_uses_template_directly_for_token_ids() -> None:
    module = _load_script()
    tokenizer = InstructTokenizer()
    _install_fakes(module, tokenizer)

    report = module.inspect("org/instruct-model", "commit-sha")

    assert report["rendered_prompt"] == "<user>attention</user><assistant>"
    assert report["rendered_token_ids"] == [101, 202, 303]
    assert report["token_count"] == 3
    assert tokenizer.calls == [(False, True), (True, True)]
    assert report["generation_config_status"] == "loaded"
    assert report["resolved_generation_config_commit"] == "generation-commit-456"
    assert report["normalized_generation_config_snapshot_source"] == (
        "GenerationConfig.to_dict() after Transformers loading; may include library "
        "defaults and runtime metadata, so its fingerprint is not a hash of raw "
        "generation_config.json bytes"
    )
    assert report["generation_protocol_contract"]["contract_fingerprint"] == (
        "sha256:acd67f36a51325fd93470524f8523b3d0d750a3260d238478c44363de4ea4442"
    )
    eos = report["generation_protocol_contract"]["special_tokens"][1]
    assert eos["field"] == "eos_token_id"
    assert eos["tokenizer_ids"] == [2]
    assert eos["generation_config_ids"] == [2, 3]
    assert eos["tokenizer_vs_generation"] == "left_strict_subset"


class InvalidTokenizingTokenizer:
    chat_template = "authored template"

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str | list[str]:
        if tokenize:
            return ["not-an-integer"]
        return "rendered"


def test_instruct_tokenizer_rejects_non_integer_token_sequence() -> None:
    module = _load_script()
    _install_fakes(module, InvalidTokenizingTokenizer())

    with pytest.raises(RuntimeError, match="flat integer token list"):
        module.inspect("org/broken-template", "commit-sha")
