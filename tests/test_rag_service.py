from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")

from about_llm.rag.ingestion import SourceDocument
from about_llm.rag.service import (
    AuthContext,
    PersistentExtractiveRAGService,
    RAGQueryRequest,
    RAGServiceConfig,
    StaticBearerAuthResolver,
    create_rag_app,
)
from about_llm.rag.sqlite_store import SQLiteChunkStore

pytestmark = [pytest.mark.contract, pytest.mark.security, pytest.mark.integration]


def _database(path: Path) -> Path:
    sources = (
        SourceDocument(
            source_id="public-security",
            tenant_id="tenant-a",
            version="v1",
            text="RAG 检索必须在排序前执行租户和主体权限过滤。",
        ),
        SourceDocument(
            source_id="engineering-citations",
            tenant_id="tenant-a",
            version="v1",
            text="RAG 生成只能使用已授权并完成引用绑定的检索证据。",
            acl=("engineering",),
        ),
        SourceDocument(
            source_id="finance-secret",
            tenant_id="tenant-a",
            version="v1",
            text="RAG 检索 权限 过滤 排序前 财务秘密 检索 权限。",
            acl=("finance",),
        ),
        SourceDocument(
            source_id="other-tenant",
            tenant_id="tenant-b",
            version="v1",
            text="RAG 检索必须在排序前执行权限过滤。",
        ),
    )
    with SQLiteChunkStore(path) as store:
        for source in sources:
            store.upsert_source(source, expected_current_version=None, max_chars=1000)
    return path


def _resolver() -> StaticBearerAuthResolver:
    return StaticBearerAuthResolver(
        {
            "engineering-token": AuthContext(
                subject_id="user-engineering",
                tenant_id="tenant-a",
                principals=("engineering",),
            ),
            "anonymous-token": AuthContext(
                subject_id="user-anonymous",
                tenant_id="tenant-a",
                principals=(),
            ),
            "empty-tenant-token": AuthContext(
                subject_id="user-empty",
                tenant_id="tenant-c",
                principals=(),
            ),
        }
    )


def _request_ids() -> Callable[[], str]:
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"request-{counter}"

    return next_id


async def _call(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


def test_query_uses_resolved_identity_and_returns_replayable_artifact(tmp_path: Path) -> None:
    database = _database(tmp_path / "rag.db")
    service = PersistentExtractiveRAGService(
        database,
        request_id_factory=_request_ids(),
    )
    app = create_rag_app(service, _resolver())

    response = asyncio.run(
        _call(
            app,
            "POST",
            "/v1/rag/query",
            headers={"Authorization": "Bearer engineering-token"},
            json={
                "query_id": "q1",
                "query": "RAG 检索为什么要在排序前做权限过滤",
                "top_k": 10,
                "budget_units": 12000,
            },
        )
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-1"
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload["request_id"] == "request-1"
    assert payload["subject_id"] == "user-engineering"
    assert payload["tenant_id"] == "tenant-a"
    assert payload["action"] == "answer"
    assert payload["artifact_fingerprint"].startswith("sha256:")
    assert payload["artifact"]["artifact_fingerprint"] == payload["artifact_fingerprint"]
    source_ids = {source["stable_source_id"] for source in payload["artifact"]["sources"]}
    assert source_ids == {"public-security", "engineering-citations"}
    assert "finance-secret" not in str(payload)
    assert "other-tenant" not in str(payload)
    for citation in payload["citations"]:
        assert citation["text"] in payload["artifact"]["rendered_context"]
        assert citation["text"] in payload["answer_text"]


def test_body_cannot_self_report_security_context_and_auth_errors_are_closed(
    tmp_path: Path,
) -> None:
    app = create_rag_app(
        PersistentExtractiveRAGService(
            _database(tmp_path / "rag.db"),
            request_id_factory=_request_ids(),
        ),
        _resolver(),
    )
    forged = asyncio.run(
        _call(
            app,
            "POST",
            "/v1/rag/query",
            headers={"Authorization": "Bearer anonymous-token"},
            json={
                "query_id": "q1",
                "query": "RAG",
                "tenant_id": "tenant-b",
                "principals": ["finance"],
            },
        )
    )
    missing = asyncio.run(
        _call(
            app,
            "POST",
            "/v1/rag/query",
            json={"query_id": "q2", "query": "RAG"},
        )
    )
    invalid = asyncio.run(
        _call(
            app,
            "POST",
            "/v1/rag/query",
            headers={"Authorization": "Bearer wrong-secret"},
            json={"query_id": "q3", "query": "RAG"},
        )
    )

    assert forged.status_code == 422
    assert forged.json()["error"]["code"] == "invalid_request"
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert missing.headers["cache-control"] == "no-store"
    assert invalid.status_code == 401
    assert "wrong-secret" not in invalid.text


def test_anonymous_and_empty_tenant_paths_cannot_see_private_documents(tmp_path: Path) -> None:
    app = create_rag_app(
        PersistentExtractiveRAGService(
            _database(tmp_path / "rag.db"),
            request_id_factory=_request_ids(),
        ),
        _resolver(),
    )
    request = {
        "query_id": "q",
        "query": "RAG 检索为什么要在排序前做权限过滤",
        "top_k": 10,
    }
    anonymous = asyncio.run(
        _call(
            app,
            "POST",
            "/v1/rag/query",
            headers={"Authorization": "Bearer anonymous-token"},
            json=request,
        )
    )
    empty = asyncio.run(
        _call(
            app,
            "POST",
            "/v1/rag/query",
            headers={"Authorization": "Bearer empty-tenant-token"},
            json=request,
        )
    )

    assert anonymous.status_code == 200
    anonymous_sources = anonymous.json()["artifact"]["sources"]
    assert [source["stable_source_id"] for source in anonymous_sources] == [
        "public-security"
    ]
    assert empty.status_code == 200
    assert empty.json()["action"] == "abstain"
    assert empty.json()["retrieved_document_ids"] == []
    assert empty.json()["artifact"]["sources"] == []


def test_service_limits_readiness_and_disabled_schema_surface(tmp_path: Path) -> None:
    database = _database(tmp_path / "rag.db")
    app = create_rag_app(
        PersistentExtractiveRAGService(
            database,
            config=RAGServiceConfig(max_top_k=2, max_budget_units=100),
            request_id_factory=_request_ids(),
        ),
        _resolver(),
    )
    live = asyncio.run(_call(app, "GET", "/health/live"))
    ready = asyncio.run(_call(app, "GET", "/health/ready"))
    schema = asyncio.run(_call(app, "GET", "/openapi.json"))
    over_limit = asyncio.run(
        _call(
            app,
            "POST",
            "/v1/rag/query",
            headers={"Authorization": "Bearer engineering-token"},
            json={"query_id": "q", "query": "RAG", "top_k": 3, "budget_units": 50},
        )
    )

    assert live.json()["status"] == "live"
    assert ready.json()["status"] == "ready"
    assert schema.status_code == 404
    assert over_limit.status_code == 422
    assert over_limit.json()["error"]["code"] == "limit_exceeded"

    database.unlink()
    not_ready = asyncio.run(_call(app, "GET", "/health/ready"))
    assert not_ready.status_code == 503
    assert not_ready.json()["status"] == "not_ready"


def test_bounded_queue_rejects_second_concurrent_request(tmp_path: Path, monkeypatch: Any) -> None:
    database = _database(tmp_path / "rag.db")
    service = PersistentExtractiveRAGService(
        database,
        config=RAGServiceConfig(
            max_concurrency=1,
            queue_timeout_seconds=0.02,
            execution_timeout_seconds=1.0,
        ),
        request_id_factory=_request_ids(),
    )
    original = service._query_sync

    def slow(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.12)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_query_sync", slow)
    app = create_rag_app(service, _resolver())

    async def scenario() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            kwargs = {
                "headers": {"Authorization": "Bearer engineering-token"},
                "json": {"query_id": "q", "query": "RAG"},
            }
            first_task = asyncio.create_task(client.post("/v1/rag/query", **kwargs))
            await asyncio.sleep(0.01)
            second = await client.post("/v1/rag/query", **kwargs)
            first = await first_task
            return first, second

    first, second = asyncio.run(scenario())
    assert first.status_code == 200
    assert second.status_code == 503
    assert second.json()["error"]["code"] == "queue_saturated"


def test_execution_timeout_and_internal_error_do_not_disclose_exception(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    database = _database(tmp_path / "rag.db")
    service = PersistentExtractiveRAGService(
        database,
        config=RAGServiceConfig(
            max_concurrency=1,
            queue_timeout_seconds=0.01,
            execution_timeout_seconds=0.02,
        ),
        request_id_factory=_request_ids(),
    )
    original = service._query_sync

    def slow(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.08)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_query_sync", slow)
    app = create_rag_app(service, _resolver())

    async def timeout_scenario() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer engineering-token"}
            timeout_response = await client.post(
                "/v1/rag/query",
                headers=headers,
                json={"query_id": "q", "query": "RAG"},
            )
            queued_response = await client.post(
                "/v1/rag/query",
                headers=headers,
                json={"query_id": "q2", "query": "RAG"},
            )
            await asyncio.sleep(0.1)
            return timeout_response, queued_response

    timeout, while_background_thread_runs = asyncio.run(timeout_scenario())
    assert timeout.status_code == 504
    assert timeout.json()["error"]["code"] == "execution_timeout"
    assert while_background_thread_runs.status_code == 503
    assert while_background_thread_runs.json()["error"]["code"] == "queue_saturated"

    def fail(*_: Any, **__: Any) -> Any:
        raise RuntimeError("C:/private/database.db secret-value")

    service.config = RAGServiceConfig(execution_timeout_seconds=1.0)
    monkeypatch.setattr(service, "_query_sync", fail)
    failed = asyncio.run(
        _call(
            app,
            "POST",
            "/v1/rag/query",
            headers={"Authorization": "Bearer engineering-token"},
            json={"query_id": "q", "query": "RAG"},
        )
    )
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "internal_error"
    assert "private" not in failed.text
    assert "secret-value" not in failed.text


@pytest.mark.parametrize(
    "config",
    [
        RAGServiceConfig(max_concurrency=1),
        RAGServiceConfig(queue_timeout_seconds=0.1),
    ],
)
def test_service_configuration_examples_are_valid(config: RAGServiceConfig) -> None:
    assert config.max_concurrency > 0


def test_request_and_config_reject_bool_or_nonfinite_limits() -> None:
    with pytest.raises(ValueError):
        RAGQueryRequest(query_id="q", query="RAG", top_k=True)
    with pytest.raises(ValueError, match="finite"):
        RAGServiceConfig(queue_timeout_seconds=float("inf"))
    with pytest.raises(TypeError, match="integers"):
        RAGServiceConfig(max_concurrency=True)
