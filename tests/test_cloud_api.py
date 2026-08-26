from __future__ import annotations

import pytest

from about_llm.integrations.cloud_api import (
    ChatMessage,
    ChatResponse,
    RequestSpec,
    build_anthropic_request,
    build_gemini_request,
    build_openai_compatible_request,
    parse_anthropic_response,
    parse_gemini_response,
    parse_openai_compatible_response,
)

pytestmark = pytest.mark.contract

MESSAGES = [
    ChatMessage("system", "Be concise."),
    ChatMessage("user", "What is RAG?"),
]


def test_openai_compatible_contract_and_redaction() -> None:
    request = build_openai_compatible_request(
        base_url="https://provider.example",
        api_key="secret",
        model="model-a",
        messages=MESSAGES,
        max_tokens=32,
    )
    assert request.url == "https://provider.example/v1/chat/completions"
    assert request.body["messages"][0] == {"role": "system", "content": "Be concise."}
    assert request.sanitized_headers()["Authorization"] == "<redacted>"
    assert "secret" not in repr(request)

    response = parse_openai_compatible_response(
        {
            "model": "model-a",
            "choices": [
                {"message": {"content": "retrieval augmented generation"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3},
        }
    )
    assert response.text == "retrieval augmented generation"
    assert response.input_tokens == 4 and response.output_tokens == 3

    with pytest.raises(ValueError, match="no text content"):
        parse_openai_compatible_response(
            {
                "choices": [
                    {"message": {"content": None, "tool_calls": []}, "finish_reason": "tool_calls"}
                ]
            }
        )


def test_anthropic_moves_system_out_of_conversation() -> None:
    request = build_anthropic_request(
        base_url="https://api.anthropic.example",
        api_key="secret",
        api_version="2023-06-01",
        model="claude-example",
        messages=MESSAGES,
        max_tokens=32,
    )
    assert request.body["system"] == "Be concise."
    assert request.body["messages"] == [{"role": "user", "content": "What is RAG?"}]
    response = parse_anthropic_response(
        {
            "model": "claude-example",
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
    )
    assert response.text == "answer" and response.finish_reason == "end_turn"


def test_gemini_maps_assistant_role_and_usage_names() -> None:
    messages = [
        ChatMessage("system", "system"),
        ChatMessage("user", "first"),
        ChatMessage("assistant", "second"),
    ]
    request = build_gemini_request(
        base_url="https://generativelanguage.example",
        api_key="secret",
        model="gemini-example",
        messages=messages,
        max_tokens=32,
    )
    assert [item["role"] for item in request.body["contents"]] == ["user", "model"]
    assert request.body["systemInstruction"]["parts"][0]["text"] == "system"
    response = parse_gemini_response(
        {
            "modelVersion": "gemini-example-001",
            "candidates": [
                {
                    "content": {"parts": [{"text": "part one"}, {"text": " part two"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4},
        }
    )
    assert response.text == "part one part two"
    assert response.input_tokens == 3 and response.output_tokens == 4


@pytest.mark.parametrize(
    "candidates, message",
    [
        ([], "exactly one candidate"),
        (
            [
                {"content": {"parts": [{"text": "first"}]}},
                {"content": {"parts": [{"text": "second"}]}},
            ],
            "exactly one candidate",
        ),
        (
            [
                {
                    "content": {
                        "parts": [
                            {"text": "I will call a tool."},
                            {"functionCall": {"name": "lookup"}},
                        ]
                    }
                }
            ],
            "non-text Gemini part",
        ),
    ],
)
def test_gemini_text_projection_rejects_lossy_candidate_shapes(
    candidates: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_gemini_response({"candidates": candidates})


def test_cloud_contracts_reject_ambiguous_coercions_and_invalid_endpoints() -> None:
    with pytest.raises(ValueError, match="absolute HTTP"):
        build_anthropic_request(
            base_url="anthropic.example",
            api_key="secret",
            api_version="2023-06-01",
            model="model",
            messages=MESSAGES,
            max_tokens=8,
        )
    with pytest.raises(ValueError, match="non-negative integers"):
        parse_openai_compatible_response(
            {
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": True},
            }
        )
    with pytest.raises(ValueError, match="non-empty string"):
        parse_anthropic_response(
            {"content": [{"type": "text", "text": 7}], "usage": {}}
        )
    with pytest.raises(ValueError, match="usageMetadata must be an object"):
        parse_gemini_response(
            {
                "candidates": [{"content": {"parts": [{"text": "answer"}]}}],
                "usageMetadata": [],
            }
        )
    with pytest.raises(ValueError, match="positive integer"):
        build_openai_compatible_request(
            base_url="https://provider.example",
            api_key="secret",
            model="model",
            messages=MESSAGES,
            max_tokens=True,
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        build_gemini_request(
            base_url="https://provider.example",
            api_key="secret",
            model="model",
            messages=MESSAGES,
            max_tokens=8,
            temperature=float("nan"),
        )


def test_direct_contract_objects_reject_boolean_usage_and_nonfinite_json() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        ChatResponse("answer", None, True, None, None)
    with pytest.raises(ValueError, match="JSON serializable"):
        RequestSpec(
            "https://provider.example/v1/chat/completions",
            {"temperature": float("inf")},
            {},
        )


def test_request_spec_rejects_case_insensitive_duplicate_header_names() -> None:
    with pytest.raises(ValueError, match="unique case-insensitively"):
        RequestSpec(
            "https://provider.example/v1/chat/completions",
            {},
            {"Authorization": "first", "authorization": "second"},
        )
