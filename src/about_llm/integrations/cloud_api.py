"""Explicit request/response contracts for major cloud model API families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {self.role}")
        if not self.content:
            raise ValueError("message content cannot be empty")


@dataclass(frozen=True)
class RequestSpec:
    url: str
    body: Mapping[str, Any]
    headers: Mapping[str, str] = field(repr=False)

    def sanitized_headers(self) -> dict[str, str]:
        sensitive = {"authorization", "x-api-key", "x-goog-api-key"}
        return {
            name: "<redacted>" if name.lower() in sensitive else value
            for name, value in self.headers.items()
        }


@dataclass(frozen=True)
class ChatResponse:
    text: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None


def _validate_generation(max_tokens: int, temperature: float) -> None:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")


def _split_system(messages: Sequence[ChatMessage]) -> tuple[str | None, list[ChatMessage]]:
    systems = [message.content for message in messages if message.role == "system"]
    conversation = [message for message in messages if message.role != "system"]
    if len(systems) > 1:
        raise ValueError("at most one system message is supported by this adapter")
    if not conversation:
        raise ValueError("at least one user or assistant message is required")
    return systems[0] if systems else None, conversation


def build_openai_compatible_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[ChatMessage],
    max_tokens: int,
    temperature: float = 0,
) -> RequestSpec:
    """Build a request for GPT or configured DeepSeek/Qwen compatible endpoints."""
    _validate_generation(max_tokens, temperature)
    if not base_url.startswith(("https://", "http://")):
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if not api_key or not model or not messages:
        raise ValueError("api_key, model and messages are required")
    return RequestSpec(
        url=base_url.rstrip("/") + "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        body={
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )


def parse_openai_compatible_response(value: Mapping[str, Any]) -> ChatResponse:
    try:
        choice = value["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("invalid OpenAI-compatible chat response") from error
    if not isinstance(text, str) or not text:
        raise ValueError("OpenAI-compatible response contains no text content")
    usage = value.get("usage") or {}
    return ChatResponse(
        text=text,
        model=_optional_str(value.get("model")),
        input_tokens=_optional_int(usage.get("prompt_tokens")),
        output_tokens=_optional_int(usage.get("completion_tokens")),
        finish_reason=_optional_str(choice.get("finish_reason")),
    )


def build_anthropic_request(
    *,
    base_url: str,
    api_key: str,
    api_version: str,
    model: str,
    messages: Sequence[ChatMessage],
    max_tokens: int,
    temperature: float = 0,
) -> RequestSpec:
    _validate_generation(max_tokens, temperature)
    system, conversation = _split_system(messages)
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": message.role, "content": message.content} for message in conversation
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system is not None:
        body["system"] = system
    return RequestSpec(
        url=base_url.rstrip("/") + "/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": api_version,
            "Content-Type": "application/json",
        },
        body=body,
    )


def parse_anthropic_response(value: Mapping[str, Any]) -> ChatResponse:
    try:
        blocks = value["content"]
        texts = [block["text"] for block in blocks if block.get("type") == "text"]
    except (KeyError, TypeError) as error:
        raise ValueError("invalid Anthropic Messages response") from error
    if not texts:
        raise ValueError("Anthropic response contains no text block")
    usage = value.get("usage") or {}
    return ChatResponse(
        text="".join(str(text) for text in texts),
        model=_optional_str(value.get("model")),
        input_tokens=_optional_int(usage.get("input_tokens")),
        output_tokens=_optional_int(usage.get("output_tokens")),
        finish_reason=_optional_str(value.get("stop_reason")),
    )


def build_gemini_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[ChatMessage],
    max_tokens: int,
    temperature: float = 0,
) -> RequestSpec:
    _validate_generation(max_tokens, temperature)
    system, conversation = _split_system(messages)
    contents = []
    for message in conversation:
        role = "model" if message.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message.content}]})
    body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if system is not None:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return RequestSpec(
        url=base_url.rstrip("/") + f"/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        body=body,
    )


def parse_gemini_response(value: Mapping[str, Any]) -> ChatResponse:
    try:
        candidate = value["candidates"][0]
        parts = candidate["content"]["parts"]
        texts = [part["text"] for part in parts if "text" in part]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("invalid Gemini generateContent response") from error
    if not texts:
        raise ValueError("Gemini response contains no text part")
    usage = value.get("usageMetadata") or {}
    return ChatResponse(
        text="".join(str(text) for text in texts),
        model=_optional_str(value.get("modelVersion")),
        input_tokens=_optional_int(usage.get("promptTokenCount")),
        output_tokens=_optional_int(usage.get("candidatesTokenCount")),
        finish_reason=_optional_str(candidate.get("finishReason")),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
