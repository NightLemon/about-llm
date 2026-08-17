from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from about_llm.inference.openai_reference import (
    CHAT_COMPLETIONS_PATH,
    HEALTH_PATH,
    MAX_REQUEST_BODY_BYTES,
    MODELS_PATH,
    ChatCompletionRequest,
    ChatMessage,
    GeneratedCompletion,
    OpenAIRequestError,
    StrictJSONBodyError,
    TransformersCPUBackend,
    build_reference_app,
    decode_strict_json_object,
    parse_chat_completion_request,
)
from about_llm.inference.sse import STREAM_FINISHED, parse_sse_data_line


class FixtureBackend:
    model_id = "fixture/model"
    backend_fingerprint = "sha256:" + "a" * 64

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def generate(self, request: ChatCompletionRequest) -> GeneratedCompletion:
        self.calls += 1
        if self.fail:
            raise RuntimeError("private backend detail")
        assert request.model == self.model_id
        return GeneratedCompletion(
            text="AB",
            text_deltas=("A", "B"),
            prompt_token_count=3,
            completion_token_ids=(4, 5),
            finish_reason="stop",
        )

    def audit_projection(self) -> Mapping[str, Any]:
        return {"implementation": "fixture", "generation_call_count": self.calls}


def valid_body(*, stream: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "model": "fixture/model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        value["stream_options"] = {"include_usage": True}
    return value


def test_strict_json_body_rejects_ambiguity_nonfinite_and_bounds() -> None:
    assert decode_strict_json_object(b'{"ok":true}') == {"ok": True}
    invalid = [
        b"",
        b"[]",
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":1e400}',
        b"\xff",
        b"{",
        b" " * (MAX_REQUEST_BODY_BYTES + 1),
    ]
    for raw in invalid:
        with pytest.raises(StrictJSONBodyError):
            decode_strict_json_object(raw)


def test_generated_completion_requires_consistent_token_and_text_projection() -> None:
    valid = GeneratedCompletion("AB", ("A", "B"), 3, (4, 5), "length")
    assert valid.completion_token_count == 2
    invalid = [
        ("", ("",), 3, (4,), "stop"),
        ("AB", ("A",), 3, (4,), "stop"),
        ("AB", ("A", "B"), 0, (4, 5), "stop"),
        ("AB", ("A", "B"), 3, (), "stop"),
        ("A", ("A", ""), 3, (4,), "stop"),
        ("A", ("A",), 3, (4,), "other"),
    ]
    for arguments in invalid:
        with pytest.raises(ValueError):
            GeneratedCompletion(*arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutate", "status", "code"),
    [
        (lambda body: body.update(extra=True), 422, "invalid_request_schema"),
        (lambda body: body.update(model="missing"), 404, "model_not_found"),
        (lambda body: body.update(temperature=0.1), 422, "unsupported_sampling"),
        (lambda body: body.update(temperature=True), 422, "unsupported_sampling"),
        (lambda body: body.update(max_tokens=0), 422, "invalid_max_tokens"),
        (lambda body: body.update(stream="yes"), 422, "invalid_stream"),
        (
            lambda body: body.update(stream_options={"include_usage": True}),
            422,
            "invalid_stream_options",
        ),
        (lambda body: body.update(messages=[]), 422, "invalid_messages"),
        (
            lambda body: body.update(messages=[{"role": "tool", "content": "x"}]),
            422,
            "invalid_messages",
        ),
        (
            lambda body: body.update(
                messages=[{"role": "assistant", "content": "x"}]
            ),
            422,
            "invalid_messages",
        ),
    ],
)
def test_request_profile_fails_closed(
    mutate: Any,
    status: int,
    code: str,
) -> None:
    body = valid_body()
    mutate(body)
    with pytest.raises(OpenAIRequestError) as captured:
        parse_chat_completion_request(
            body,
            model_id="fixture/model",
            maximum_new_tokens=2,
        )
    assert captured.value.status_code == status
    assert captured.value.code == code


def test_stream_profile_requires_explicit_true_usage_option() -> None:
    body = valid_body(stream=True)
    body["stream_options"] = {"include_usage": False}
    with pytest.raises(OpenAIRequestError, match="include_usage"):
        parse_chat_completion_request(
            body,
            model_id="fixture/model",
            maximum_new_tokens=2,
        )


def test_reference_app_executes_nonstream_stream_and_safe_errors() -> None:
    async def exercise() -> None:
        backend = FixtureBackend()
        app = build_reference_app(
            backend,
            bearer_token="t" * 32,
            maximum_new_tokens=2,
        )
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        auth = {"Authorization": "Bearer " + "t" * 32}
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            unauthorized = await client.get(HEALTH_PATH)
            duplicate_auth = await client.get(
                HEALTH_PATH,
                headers=[
                    ("Authorization", "Bearer " + "t" * 32),
                    ("Authorization", "Bearer " + "t" * 32),
                ],
            )
            assert unauthorized.status_code == duplicate_auth.status_code == 401
            assert unauthorized.json()["error"]["code"] == "invalid_api_key"

            health = await client.get(HEALTH_PATH, headers=auth)
            models = await client.get(MODELS_PATH, headers=auth)
            assert health.json()["status"] == "ready"
            assert models.json()["data"][0]["id"] == "fixture/model"

            wrong_media = await client.post(
                CHAT_COMPLETIONS_PATH,
                headers={**auth, "Content-Type": "text/plain"},
                content=b"{}",
            )
            invalid_json = await client.post(
                CHAT_COMPLETIONS_PATH,
                headers={**auth, "Content-Type": "application/json"},
                content=b'{"x":1,"x":2}',
            )
            oversized = await client.post(
                CHAT_COMPLETIONS_PATH,
                headers={**auth, "Content-Type": "application/json"},
                content=b" " * (MAX_REQUEST_BODY_BYTES + 1),
            )
            assert wrong_media.status_code == 415
            assert invalid_json.status_code == 400
            assert oversized.status_code == 413

            nonstream = await client.post(
                CHAT_COMPLETIONS_PATH,
                headers=auth,
                json=valid_body(),
            )
            assert nonstream.status_code == 200
            payload = nonstream.json()
            assert payload["choices"] == [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "AB"},
                    "finish_reason": "stop",
                }
            ]
            assert payload["usage"] == {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            }

            streamed = await client.post(
                CHAT_COMPLETIONS_PATH,
                headers=auth,
                json=valid_body(stream=True),
            )
            assert streamed.headers["content-type"].startswith("text/event-stream")
            content = ""
            usage: dict[str, Any] | None = None
            finished = False
            for line in streamed.text.splitlines(keepends=True):
                event = parse_sse_data_line(line)
                if event is None:
                    continue
                if event is STREAM_FINISHED:
                    finished = True
                    continue
                choices = event.get("choices") or []
                if choices:
                    content += choices[0].get("delta", {}).get("content", "")
                if event.get("usage") is not None:
                    usage = event["usage"]
            assert finished and content == "AB"
            assert usage == {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            }
            assert app.state.reference_service.audit_projection() == {
                "service_version": "about-llm.openai-reference-service.v1",
                "accepted_requests": 2,
                "stream_requests": 1,
                "nonstream_requests": 1,
                "failed_backend_requests": 0,
                "single_process_admission_limit": 1,
                "backend": {
                    "implementation": "fixture",
                    "generation_call_count": 2,
                },
            }

        failing_app = build_reference_app(
            FixtureBackend(fail=True),
            bearer_token="t" * 32,
            maximum_new_tokens=2,
        )
        failing_transport = httpx.ASGITransport(
            app=failing_app, raise_app_exceptions=False
        )
        async with httpx.AsyncClient(
            transport=failing_transport,
            base_url="http://testserver",
        ) as client:
            failed = await client.post(
                CHAT_COMPLETIONS_PATH,
                headers=auth,
                json=valid_body(),
            )
        assert failed.status_code == 500
        assert failed.json()["error"]["message"] == "Model backend failed"
        assert "private backend detail" not in failed.text

    asyncio.run(exercise())


def test_transformers_cpu_backend_executes_generate_and_audits_identity() -> None:
    import torch

    class TinyTokenizer:
        eos_token_id = 4
        pad_token_id = 0
        chat_template = "fixture"

        def __len__(self) -> int:
            return 10

        def apply_chat_template(self, *_: Any, **__: Any) -> Any:
            return torch.tensor([[1, 2]], dtype=torch.long)

        def decode(self, ids: Any, **_: Any) -> str:
            mapping = {3: "A", 4: "B"}
            return "".join(mapping[int(value)] for value in ids)

    class TinyModel:
        def __init__(self) -> None:
            self.generate_calls = 0

        def to(self, _: str) -> TinyModel:
            return self

        def requires_grad_(self, _: bool) -> TinyModel:
            return self

        def eval(self) -> TinyModel:
            return self

        def generate(self, **_: Any) -> Any:
            self.generate_calls += 1
            return SimpleNamespace(
                sequences=torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            )

    model = TinyModel()
    backend = TransformersCPUBackend(
        model_id="fixture/model",
        backend_fingerprint="sha256:" + "b" * 64,
        model=model,
        tokenizer=TinyTokenizer(),
        maximum_prompt_tokens=8,
    )
    request = ChatCompletionRequest(
        model="fixture/model",
        messages=(ChatMessage("user", "hello"),),
        max_tokens=2,
        stream=False,
        include_usage=False,
    )
    result = asyncio.run(backend.generate(request))
    assert result == GeneratedCompletion("AB", ("A", "B"), 2, (3, 4), "stop")
    assert model.generate_calls == 1
    audit = backend.audit_projection()
    assert audit["generation_call_count"] == 1
    assert audit["last_execution"] == {
        "prompt_token_count": 2,
        "completion_token_ids": [3, 4],
        "completion_text_fingerprint": (
            "sha256:4f2d16fe91c2f8979725754479e1640a0a81823b3407e9e196bcb97e6a806ec0"
        ),
        "finish_reason": "stop",
    }


def test_reference_service_configuration_fails_closed() -> None:
    backend = FixtureBackend()
    with pytest.raises(ValueError, match="bearer token"):
        build_reference_app(backend, bearer_token="short", maximum_new_tokens=2)
    with pytest.raises(ValueError, match="positive"):
        build_reference_app(backend, bearer_token="t" * 32, maximum_new_tokens=0)
    backend.backend_fingerprint = "invalid"
    with pytest.raises(ValueError, match="identity"):
        build_reference_app(backend, bearer_token="t" * 32, maximum_new_tokens=2)


def test_openai_error_does_not_echo_rejected_model() -> None:
    rejected = "PRIVATE-MODEL-NAME"
    body = valid_body()
    body["model"] = rejected
    with pytest.raises(OpenAIRequestError) as captured:
        parse_chat_completion_request(
            body,
            model_id="fixture/model",
            maximum_new_tokens=2,
        )
    serialized = json.dumps(
        {"code": captured.value.code, "message": captured.value.safe_message}
    )
    assert rejected not in serialized
