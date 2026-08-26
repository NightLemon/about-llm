"""Explicit request/response contracts for major cloud model API families."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

Role = Literal["system", "user", "assistant"]
CREDENTIAL_HEADER_NAMES = frozenset(
    {"authorization", "x-api-key", "x-goog-api-key"}
)


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {self.role}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("message content cannot be empty")


@dataclass(frozen=True)
class RequestSpec:
    url: str
    body: Mapping[str, Any]
    headers: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        _validate_base_url(self.url)
        if not all(
            isinstance(name, str)
            and name.strip()
            and isinstance(value, str)
            and value
            for name, value in self.headers.items()
        ):
            raise ValueError("request headers must contain non-empty string names and values")
        normalized_names = [name.lower() for name in self.headers]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("request header names must be unique case-insensitively")
        try:
            body_snapshot = json.loads(
                json.dumps(self.body, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("request body must be JSON serializable") from error
        if not isinstance(body_snapshot, dict):
            raise ValueError("request body must be a JSON object")
        object.__setattr__(self, "body", MappingProxyType(body_snapshot))
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))

    def sanitized_headers(self) -> dict[str, str]:
        return {
            name: (
                "<redacted>"
                if name.lower() in CREDENTIAL_HEADER_NAMES
                else value
            )
            for name, value in self.headers.items()
        }


@dataclass(frozen=True)
class ChatResponse:
    text: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("response text cannot be empty")
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model
        ):
            raise ValueError("model must be a non-empty string when present")
        if self.finish_reason is not None and (
            not isinstance(self.finish_reason, str) or not self.finish_reason
        ):
            raise ValueError("finish_reason must be a non-empty string when present")


def _validate_generation(max_tokens: int, temperature: float) -> None:
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be a finite non-negative number")


def _validate_base_url(value: str) -> None:
    if not isinstance(value, str) or not value.startswith(("https://", "http://")):
        raise ValueError("base_url must be an absolute HTTP(S) URL")


def _validate_required(**values: str) -> None:
    missing = [
        name
        for name, value in values.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if missing:
        raise ValueError(f"required non-empty field(s): {', '.join(missing)}")


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
    _validate_base_url(base_url)
    _validate_required(api_key=api_key, model=model)
    if not messages:
        raise ValueError("at least one message is required")
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
        choices = value["choices"]
        choice = choices[0]
        message = choice["message"]
        text = message["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("invalid OpenAI-compatible chat response") from error
    if not isinstance(choice, Mapping) or not isinstance(message, Mapping):
        raise ValueError("invalid OpenAI-compatible chat response")
    if not isinstance(text, str) or not text:
        raise ValueError("OpenAI-compatible response contains no text content")
    usage = _optional_mapping(value.get("usage"), label="OpenAI-compatible usage")
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
    _validate_base_url(base_url)
    _validate_required(api_key=api_key, api_version=api_version, model=model)
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
    except (KeyError, TypeError) as error:
        raise ValueError("invalid Anthropic Messages response") from error
    if not isinstance(blocks, list) or not all(isinstance(block, Mapping) for block in blocks):
        raise ValueError("invalid Anthropic Messages response")
    texts = [block.get("text") for block in blocks if block.get("type") == "text"]
    if not texts:
        raise ValueError("Anthropic response contains no text block")
    if not all(isinstance(text, str) and text for text in texts):
        raise ValueError("Anthropic text block must contain non-empty string text")
    usage = _optional_mapping(value.get("usage"), label="Anthropic usage")
    return ChatResponse(
        text="".join(cast(str, text) for text in texts),
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
    _validate_base_url(base_url)
    _validate_required(api_key=api_key, model=model)
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
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError(
            "Gemini text response requires exactly one candidate; "
            "inspect promptFeedback for a blocked prompt"
        )
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise ValueError("invalid Gemini generateContent response")
    content = candidate.get("content")
    if not isinstance(content, Mapping):
        raise ValueError("invalid Gemini generateContent response")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Gemini text response requires at least one part")
    texts: list[str] = []
    for part in parts:
        if not isinstance(part, Mapping):
            raise ValueError("invalid Gemini generateContent response")
        if set(part) != {"text"}:
            raise ValueError("non-text Gemini part is outside this adapter")
        text = part.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Gemini text part must contain non-empty string text")
        texts.append(text)
    usage = _optional_mapping(value.get("usageMetadata"), label="Gemini usageMetadata")
    return ChatResponse(
        text="".join(texts),
        model=_optional_str(value.get("modelVersion")),
        input_tokens=_optional_int(usage.get("promptTokenCount")),
        output_tokens=_optional_int(usage.get("candidatesTokenCount")),
        finish_reason=_optional_str(candidate.get("finishReason")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("token usage values must be non-negative integers")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("optional response fields must be non-empty strings")
    return value


def _optional_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value
