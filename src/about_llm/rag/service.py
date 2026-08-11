"""FastAPI boundary for the persistent, deterministic extractive RAG baseline.

The service deliberately does not accept tenant or principal identity in the
JSON body.  A caller-supplied authentication resolver establishes the trusted
security context before SQLite visibility filtering and BM25 scoring.
"""

from __future__ import annotations

import asyncio
import hmac
import math
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from about_llm.rag.bm25 import BM25Index
from about_llm.rag.context_packing import utf8_byte_length
from about_llm.rag.extractive import ExtractiveAnswerConfig, generate_extractive_answer
from about_llm.rag.ingestion import SourceChunk
from about_llm.rag.models import Document, SearchResult
from about_llm.rag.sqlite_store import SQLiteChunkStore

RAG_SERVICE_REVISION = "about-llm.rag-extractive-asgi.v1"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_EVIDENCE_BOUNDARY = (
    "persistent SQLite + authorization-first BM25 + deterministic exact-span baseline; "
    "no learned retriever/reranker, LLM generation, remote auth, network transport, "
    "multi-process admission control, or production SLO is proved"
)


@dataclass(frozen=True)
class AuthContext:
    """Trusted identity returned by an authentication adapter."""

    subject_id: str
    tenant_id: str
    principals: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.subject_id.strip() or not self.tenant_id.strip():
            raise ValueError("subject_id and tenant_id must be non-empty")
        if any(not principal.strip() for principal in self.principals):
            raise ValueError("principals must contain non-empty strings")
        if len(self.principals) != len(set(self.principals)):
            raise ValueError("principals cannot contain duplicates")


class AuthResolver(Protocol):
    """Resolve a transport credential into a trusted application identity."""

    def resolve(self, authorization: str | None) -> AuthContext: ...


class RAGServiceError(Exception):
    """Typed client-visible failure with a stable code and HTTP status."""

    def __init__(self, *, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class StaticBearerAuthResolver:
    """Small demo/test resolver; production must replace it with real authentication."""

    def __init__(self, identities: Mapping[str, AuthContext]) -> None:
        if not identities:
            raise ValueError("at least one bearer identity is required")
        copied: dict[str, AuthContext] = {}
        for token, context in identities.items():
            if not isinstance(token, str) or not token or token.strip() != token:
                raise ValueError("bearer tokens must be non-empty strings without edge spaces")
            if any(character.isspace() for character in token):
                raise ValueError("bearer tokens cannot contain whitespace")
            copied[token] = context
        self._identities = copied

    def resolve(self, authorization: str | None) -> AuthContext:
        if authorization is None:
            raise RAGServiceError(
                code="unauthorized",
                message="a bearer credential is required",
                status_code=401,
            )
        scheme, separator, supplied = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer" or not supplied:
            raise RAGServiceError(
                code="unauthorized",
                message="authorization must use one Bearer credential",
                status_code=401,
            )
        if supplied.strip() != supplied or any(character.isspace() for character in supplied):
            raise RAGServiceError(
                code="unauthorized",
                message="authorization must use one Bearer credential",
                status_code=401,
            )
        matched: AuthContext | None = None
        for expected, context in self._identities.items():
            if hmac.compare_digest(supplied, expected):
                matched = context
        if matched is None:
            raise RAGServiceError(
                code="unauthorized",
                message="the bearer credential is invalid",
                status_code=401,
            )
        return matched


@dataclass(frozen=True)
class RAGServiceConfig:
    max_query_chars: int = 4000
    max_top_k: int = 20
    max_budget_units: int = 12000
    max_chunks_per_source: int = 2
    max_concurrency: int = 8
    queue_timeout_seconds: float = 0.25
    execution_timeout_seconds: float = 3.0
    sqlite_busy_timeout_ms: int = 1000

    def __post_init__(self) -> None:
        integer_fields = (
            self.max_query_chars,
            self.max_top_k,
            self.max_budget_units,
            self.max_chunks_per_source,
            self.max_concurrency,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise TypeError("service limits must be integers")
        if any(value <= 0 for value in integer_fields):
            raise ValueError("service limits must be positive")
        if (
            isinstance(self.sqlite_busy_timeout_ms, bool)
            or not isinstance(self.sqlite_busy_timeout_ms, int)
            or self.sqlite_busy_timeout_ms < 0
        ):
            raise ValueError("sqlite_busy_timeout_ms must be a non-negative integer")
        for name in ("queue_timeout_seconds", "execution_timeout_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite duration")


class RAGQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=100)
    budget_units: int = Field(default=12000, ge=1, le=1_000_000)


class CitationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    short_source_id: str
    stable_source_id: str
    document_id: str
    start_char: int
    end_char: int
    text: str


class RAGQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    service_revision: str
    request_id: str
    query_id: str
    subject_id: str
    tenant_id: str
    action: str
    answer_text: str
    citations: list[CitationResponse]
    retrieved_document_ids: list[str]
    artifact_fingerprint: str
    artifact: dict[str, Any]
    evidence_boundary: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    service_revision: str
    status: str


RequestIdFactory = Callable[[], str]


class PersistentExtractiveRAGService:
    """Bounded async facade over request-local SQLite/BM25/extractive execution."""

    def __init__(
        self,
        database_path: Path,
        *,
        config: RAGServiceConfig | None = None,
        request_id_factory: RequestIdFactory | None = None,
    ) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be a pathlib.Path")
        if not database_path.is_file():
            raise ValueError("database_path must identify an existing SQLite file")
        self.database_path = database_path
        self.config = config or RAGServiceConfig()
        self._request_id_factory = request_id_factory or (lambda: str(uuid.uuid4()))
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._validate_database()

    def issue_request_id(self) -> str:
        request_id = self._request_id_factory()
        if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
            raise RuntimeError("request id factory returned an invalid value")
        return request_id

    def ready(self) -> bool:
        try:
            self._validate_database()
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    async def query(
        self,
        request: RAGQueryRequest,
        auth: AuthContext,
        *,
        request_id: str,
    ) -> RAGQueryResponse:
        self._validate_request(request)
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.queue_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise RAGServiceError(
                code="queue_saturated",
                message="the bounded service queue did not admit the request in time",
                status_code=503,
            ) from error
        release_on_exit = True
        try:
            work = asyncio.create_task(
                asyncio.to_thread(self._query_sync, request, auth, request_id)
            )
            try:
                return await asyncio.wait_for(
                    asyncio.shield(work),
                    timeout=self.config.execution_timeout_seconds,
                )
            except asyncio.TimeoutError as error:
                release_on_exit = False
                work.add_done_callback(self._release_after_background_work)
                raise RAGServiceError(
                    code="execution_timeout",
                    message="the request exceeded the service execution deadline",
                    status_code=504,
                ) from error
            except asyncio.CancelledError:
                release_on_exit = False
                work.add_done_callback(self._release_after_background_work)
                raise
        finally:
            if release_on_exit:
                self._semaphore.release()

    def _release_after_background_work(
        self,
        work: asyncio.Task[RAGQueryResponse],
    ) -> None:
        try:
            work.exception()
        except asyncio.CancelledError:
            pass
        finally:
            self._semaphore.release()

    def _validate_database(self) -> None:
        if not self.database_path.is_file():
            raise ValueError("database file is unavailable")
        with SQLiteChunkStore(
            self.database_path,
            busy_timeout_ms=self.config.sqlite_busy_timeout_ms,
        ):
            pass

    def _validate_request(self, request: RAGQueryRequest) -> None:
        if request.query_id.strip() != request.query_id or not request.query_id.strip():
            raise RAGServiceError(
                code="invalid_request",
                message="query_id must be non-empty without edge whitespace",
                status_code=422,
            )
        if not request.query.strip():
            raise RAGServiceError(
                code="invalid_request",
                message="query must contain non-whitespace text",
                status_code=422,
            )
        if len(request.query) > self.config.max_query_chars:
            raise RAGServiceError(
                code="limit_exceeded",
                message="query exceeds the configured character limit",
                status_code=422,
            )
        if request.top_k > self.config.max_top_k:
            raise RAGServiceError(
                code="limit_exceeded",
                message="top_k exceeds the configured service limit",
                status_code=422,
            )
        if request.budget_units > self.config.max_budget_units:
            raise RAGServiceError(
                code="limit_exceeded",
                message="budget_units exceeds the configured service limit",
                status_code=422,
            )

    def _query_sync(
        self,
        request: RAGQueryRequest,
        auth: AuthContext,
        request_id: str,
    ) -> RAGQueryResponse:
        with SQLiteChunkStore(
            self.database_path,
            busy_timeout_ms=self.config.sqlite_busy_timeout_ms,
        ) as store:
            chunks = store.visible_chunks(
                tenant_id=auth.tenant_id,
                principals=auth.principals,
            )
        documents = tuple(_chunk_document(chunk) for chunk in chunks)
        results: tuple[SearchResult, ...]
        if documents:
            results = tuple(
                BM25Index(documents).search(
                    request.query,
                    tenant_id=auth.tenant_id,
                    principals=auth.principals,
                    top_k=request.top_k,
                )
            )
        else:
            results = ()
        artifact = generate_extractive_answer(
            results,
            query_id=request.query_id,
            query=request.query,
            tenant_id=auth.tenant_id,
            principals=auth.principals,
            cost_fn=utf8_byte_length,
            budget_units=request.budget_units,
            cost_unit="utf8_bytes",
            max_chunks_per_source=self.config.max_chunks_per_source,
            config=ExtractiveAnswerConfig(),
        )
        citations = [
            CitationResponse(
                short_source_id=span.short_source_id,
                stable_source_id=span.stable_source_id,
                document_id=span.document_id,
                start_char=span.start_char,
                end_char=span.end_char,
                text=span.text,
            )
            for span in artifact.proposed_spans
        ]
        return RAGQueryResponse(
            service_revision=RAG_SERVICE_REVISION,
            request_id=request_id,
            query_id=request.query_id,
            subject_id=auth.subject_id,
            tenant_id=auth.tenant_id,
            action=artifact.action.value,
            answer_text=artifact.answer_text,
            citations=citations,
            retrieved_document_ids=[result.document.document_id for result in results],
            artifact_fingerprint=artifact.artifact_fingerprint,
            artifact=artifact.to_dict(),
            evidence_boundary=_EVIDENCE_BOUNDARY,
        )


def _chunk_document(chunk: SourceChunk) -> Document:
    metadata = dict(chunk.metadata)
    metadata.update(
        {
            "source_id": chunk.source_id,
            "source_version": chunk.source_version,
            "heading_path": chunk.heading_path,
            "ordinal": chunk.ordinal,
            "content_hash": chunk.content_hash,
        }
    )
    return Document(
        document_id=chunk.chunk_id,
        text=chunk.text,
        tenant_id=chunk.tenant_id,
        metadata=metadata,
        acl=chunk.acl,
    )


def create_rag_app(
    service: PersistentExtractiveRAGService,
    auth_resolver: AuthResolver,
) -> FastAPI:
    """Create a closed-schema app with no interactive docs or default auth."""
    app = FastAPI(
        title="About LLM deterministic RAG service",
        version=RAG_SERVICE_REVISION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def request_id(request: Request) -> str:
        existing = getattr(request.state, "request_id", None)
        if isinstance(existing, str):
            return existing
        created = service.issue_request_id()
        request.state.request_id = created
        return created

    def error_response(
        request: Request,
        *,
        status_code: int,
        code: str,
        message: str,
    ) -> JSONResponse:
        identifier = request_id(request)
        headers = {"X-Request-ID": identifier, "Cache-Control": "no-store"}
        if status_code == 401:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse(
            status_code=status_code,
            headers=headers,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "request_id": identifier,
                }
            },
        )

    async def validation_handler(request: Request, _: Exception) -> JSONResponse:
        return error_response(
            request,
            status_code=422,
            code="invalid_request",
            message="request JSON does not satisfy the closed query schema",
        )

    async def service_error_handler(request: Request, error: Exception) -> JSONResponse:
        if not isinstance(error, RAGServiceError):
            return error_response(
                request,
                status_code=500,
                code="internal_error",
                message="the service could not complete the request",
            )
        return error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
        )

    async def internal_error_handler(request: Request, _: Exception) -> JSONResponse:
        return error_response(
            request,
            status_code=500,
            code="internal_error",
            message="the service could not complete the request",
        )

    app.add_exception_handler(RequestValidationError, validation_handler)
    app.add_exception_handler(RAGServiceError, service_error_handler)
    app.add_exception_handler(Exception, internal_error_handler)

    @app.get("/health/live", response_model=HealthResponse)
    async def live(response: Response) -> HealthResponse:
        response.headers["Cache-Control"] = "no-store"
        return HealthResponse(service_revision=RAG_SERVICE_REVISION, status="live")

    @app.get("/health/ready", response_model=HealthResponse)
    async def ready(response: Response) -> HealthResponse:
        response.headers["Cache-Control"] = "no-store"
        if not service.ready():
            response.status_code = 503
            return HealthResponse(service_revision=RAG_SERVICE_REVISION, status="not_ready")
        return HealthResponse(service_revision=RAG_SERVICE_REVISION, status="ready")

    @app.post("/v1/rag/query", response_model=RAGQueryResponse)
    async def query_rag(
        payload: RAGQueryRequest,
        request: Request,
        response: Response,
    ) -> RAGQueryResponse:
        identifier = request_id(request)
        auth = auth_resolver.resolve(request.headers.get("authorization"))
        result = await service.query(payload, auth, request_id=identifier)
        response.headers["X-Request-ID"] = identifier
        response.headers["Cache-Control"] = "no-store"
        return result

    return app
