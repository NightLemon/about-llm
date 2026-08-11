from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from about_llm.finetuning.data import (
    ChatMessage,
    DataSplit,
    MessageRole,
    SFTRecord,
)
from about_llm.finetuning.template import audit_assistant_masks

RENDERER_IDENTITY = {"tokenizer": "fixture@v1", "chat_template": "fixture-template"}


def _record() -> SFTRecord:
    return SFTRecord(
        "case-1",
        (
            ChatMessage(MessageRole.USER, "question"),
            ChatMessage(MessageRole.ASSISTANT, "answer"),
        ),
        "unit-test",
        "test-only",
        "definition",
        "en",
        "normal",
        "group-1",
        DataSplit.TRAIN,
    )


def _renderer(
    input_ids: object = (10, 11, 12, 13),
    assistant_masks: object = (0, 0, 1, 1),
) -> Any:
    def render(messages: list[dict[str, str]]) -> Mapping[str, Any]:
        assert messages[-1]["role"] == "assistant"
        return {"input_ids": input_ids, "assistant_masks": assistant_masks}

    return render


def test_mask_audit_reports_explicit_unit_and_scope() -> None:
    report = audit_assistant_masks(
        (_record(),),
        render=_renderer(),
        renderer_identity=RENDERER_IDENTITY,
        max_length=8,
    )
    payload = report.to_dict()

    assert report.record_count == 1
    assert report.input_token_count == 4
    assert report.assistant_token_count == 2
    assert payload["gate_passed"] is True
    assert payload["samples"] == [
        {
            "record_id": "case-1",
            "input_token_count": 4,
            "assistant_token_count": 2,
            "token_mask_fingerprint": payload["samples"][0]["token_mask_fingerprint"],
        }
    ]
    assert payload["scope"] == {
        "target_tokenizer_executed": True,
        "tokenizer_reported_assistant_mask_checked": True,
        "right_truncation_rejected": True,
        "collator_labels_verified": False,
        "mask_semantics_independently_verified": False,
    }
    assert payload["renderer_identity"] == RENDERER_IDENTITY
    assert payload["renderer_fingerprint"].startswith("sha256:")
    assert payload["ordered_dataset_fingerprint"].startswith("sha256:")
    assert payload["manifest_fingerprint"].startswith("sha256:")


@pytest.mark.parametrize(
    ("renderer", "expected"),
    [
        (_renderer(assistant_masks=None), "assistant_masks must be an integer sequence"),
        (_renderer(assistant_masks=(0, 0, 0, 0)), "reported no assistant tokens"),
        (_renderer(assistant_masks=(0, 1)), "must have equal lengths"),
        (_renderer(assistant_masks=(0, 0, 1, 2)), "must contain only 0 or 1"),
        (_renderer(input_ids=(10, -1), assistant_masks=(0, 1)), "non-negative ids"),
        (_renderer(input_ids=(10, True), assistant_masks=(0, 1)), "only integers"),
    ],
)
def test_mask_audit_rejects_malformed_or_empty_masks(
    renderer: Any, expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        audit_assistant_masks(
            (_record(),),
            render=renderer,
            renderer_identity=RENDERER_IDENTITY,
            max_length=8,
        )


def test_mask_audit_rejects_silent_right_truncation() -> None:
    with pytest.raises(ValueError, match="explicit preprocessing"):
        audit_assistant_masks(
            (_record(),),
            render=_renderer(),
            renderer_identity=RENDERER_IDENTITY,
            max_length=3,
        )


def test_mask_audit_attributes_template_failures_to_record() -> None:
    def broken(messages: list[dict[str, str]]) -> Mapping[str, Any]:
        raise RuntimeError("missing generation marker")

    with pytest.raises(ValueError, match=r"case-1.*missing generation marker"):
        audit_assistant_masks(
            (_record(),),
            render=broken,
            renderer_identity=RENDERER_IDENTITY,
            max_length=8,
        )


@pytest.mark.parametrize("max_length", [0, -1, True, 1.5])
def test_mask_audit_requires_positive_integer_max_length(max_length: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        audit_assistant_masks(
            (_record(),),
            render=_renderer(),
            renderer_identity=RENDERER_IDENTITY,
            max_length=max_length,
        )


def test_mask_audit_requires_records_and_callable_renderer() -> None:
    with pytest.raises(ValueError, match="at least one"):
        audit_assistant_masks(
            (),
            render=_renderer(),
            renderer_identity=RENDERER_IDENTITY,
            max_length=8,
        )


def test_mask_manifest_changes_with_renderer_or_tokenization() -> None:
    first = audit_assistant_masks(
        (_record(),),
        render=_renderer(),
        renderer_identity=RENDERER_IDENTITY,
        max_length=8,
    )
    changed_renderer = audit_assistant_masks(
        (_record(),),
        render=_renderer(),
        renderer_identity={"tokenizer": "fixture@v2"},
        max_length=8,
    )
    changed_tokens = audit_assistant_masks(
        (_record(),),
        render=_renderer(input_ids=(20, 21, 22, 23)),
        renderer_identity=RENDERER_IDENTITY,
        max_length=8,
    )

    assert first.renderer_fingerprint != changed_renderer.renderer_fingerprint
    assert first.manifest_fingerprint != changed_renderer.manifest_fingerprint
    assert first.samples[0].token_mask_fingerprint != (
        changed_tokens.samples[0].token_mask_fingerprint
    )
    assert first.manifest_fingerprint != changed_tokens.manifest_fingerprint


@pytest.mark.parametrize("identity", [{}, {"temperature": float("nan")}])
def test_mask_audit_rejects_unusable_renderer_identity(identity: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="renderer_identity"):
        audit_assistant_masks(
            (_record(),),
            render=_renderer(),
            renderer_identity=identity,
            max_length=8,
        )


def test_real_transformers_template_returns_generation_mask() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                "[UNK]": 0,
                "</s>": 1,
                "user": 2,
                ":": 3,
                "question": 4,
                "assistant": 5,
                "answer": 6,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="</s>",
        pad_token="</s>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] + ': ' }}"
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] }}{% endgeneration %}"
        "{% else %}{{ message['content'] }}{% endif %}"
        "{{ eos_token }}{% endfor %}"
    )

    report = audit_assistant_masks(
        (_record(),),
        render=lambda messages: tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        ),
        renderer_identity={
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template": tokenizer.chat_template,
        },
        max_length=32,
    )

    assert report.assistant_token_count > 0
    assert report.samples[0].assistant_token_count < report.input_token_count
    with pytest.raises(ValueError, match="render must be callable"):
        audit_assistant_masks(
            (_record(),),
            render=None,  # type: ignore[arg-type]
            renderer_identity=RENDERER_IDENTITY,
            max_length=8,
        )
