from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from ai_pm_contracts import (
    AnswerProcessPhase,
    AnswerProcessTrace,
    AnswerRequest,
    ApiError,
    ApiErrorResponse,
    FactBuildManifest,
    SourceWindowRequest,
    SourceWindowResponse,
    SseDoneEvent,
    SseErrorEvent,
    SseEvidenceEvent,
    SseEvent,
    SsePhaseEvent,
    SseRetrievalSummaryEvent,
)
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from retrieval_build import FactStore, load_and_validate_bundle_manifest
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .conversations import Conversation, ConversationStore, Message
from .settings import Settings

_SSE_ADAPTER = TypeAdapter(SseEvent)


class AnswerService(Protocol):
    """Injection boundary for the later direct/agentic answer orchestrators."""

    def stream_answer(
        self,
        *,
        conversation_id: str,
        request: AnswerRequest,
        history: Sequence[Message],
    ) -> AsyncIterator[SseEvent]: ...


class FactReader(Protocol):
    def read_source_window(self, locator: Any, window: int = 0) -> tuple[Any, ...]: ...


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str | None = Field(default=None, max_length=200)


class ConversationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=200)


def _api_error(
    status_code: int,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    body = ApiErrorResponse(
        error=ApiError(
            code=code,
            message=message,
            retryable=retryable,
            details=details or {},
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _sse(event: SseEvent) -> bytes:
    validated = _SSE_ADAPTER.validate_python(event)
    payload = json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"data: {payload}\n\n".encode()


def _error_from_exception(exc: Exception) -> ApiError:
    code = getattr(exc, "code", None)
    return ApiError(
        code=str(code) if code else "ANSWER_SERVICE_ERROR",
        message=str(exc) or type(exc).__name__,
        retryable=bool(getattr(exc, "retryable", False)),
    )


def _read_health(
    settings: Settings, answer_service: AnswerService | None = None
) -> dict[str, Any]:
    facts_ready = False
    lexical_ready = False
    semantic_ready = False
    document_count = 0
    chunk_count = 0
    detail: dict[str, Any] = {}

    facts_manifest = settings.resolved_facts_root / "manifests" / "build.json"
    try:
        manifest = FactBuildManifest.model_validate_json(facts_manifest.read_bytes())
        facts_ready = True
        document_count = manifest.document_count
        chunk_count = manifest.chunk_count
    except (OSError, ValueError) as exc:
        detail["facts"] = str(exc)

    try:
        bundle = load_and_validate_bundle_manifest(
            settings.resolved_retrieval_manifest_path
        )
        lexical_ready = bundle.lexical_status == "ready"
        # test_only artifacts are intentionally not production-ready.
        semantic_ready = bundle.semantic_status == "ready"
        if bundle.semantic_reason:
            detail["semantic_reason"] = bundle.semantic_reason
        detail["retrieval_build_id"] = bundle.build_id
    except (OSError, RuntimeError, ValueError) as exc:
        detail["retrieval"] = str(exc)

    deepseek_ready = bool(settings.deepseek_api_key)
    zhipu_ready = bool(settings.zhipu_api_key)
    local_ready = facts_ready and lexical_ready
    answer_ready = local_ready and deepseek_ready
    full_ready = answer_ready and semantic_ready and zhipu_ready
    return {
        "status": "ready" if full_ready else "answer_ready" if answer_ready else "degraded",
        "agentic_enabled": bool(
            answer_service is not None
            and getattr(answer_service, "agentic_available", False)
        ),
        "facts_ready": facts_ready,
        "lexical_ready": lexical_ready,
        "semantic_ready": semantic_ready,
        "deepseek_ready": deepseek_ready,
        "zhipu_ready": zhipu_ready,
        "answer_ready": answer_ready,
        "full_ready": full_ready,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "detail": detail,
    }


def create_app(
    settings: Settings | None = None,
    *,
    conversation_store: ConversationStore | None = None,
    answer_service: AnswerService | None = None,
    fact_reader: FactReader | None = None,
    trusted_hosts: Sequence[str] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    store = conversation_store or ConversationStore(settings.resolved_chat_db_path)
    if answer_service is None:
        answer_service = _build_default_answer_service(settings)
    app = FastAPI(title="Lenny API", version="0.1.0")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(trusted_hosts or ("localhost", "127.0.0.1")),
    )
    app.state.settings = settings
    app.state.conversation_store = store
    app.state.answer_service = answer_service
    app.state.fact_reader = fact_reader
    app.state.llm_wiki_adapter = _build_llm_wiki_adapter(settings)

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return _read_health(settings, app.state.answer_service)

    @app.get("/api/v1/index/status")
    def index_status() -> Any:
        try:
            bundle = load_and_validate_bundle_manifest(
                settings.resolved_retrieval_manifest_path
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return _api_error(
                503,
                str(getattr(exc, "code", "INDEX_NOT_READY")),
                str(exc),
                retryable=True,
            )
        return bundle.model_dump(mode="json")

    @app.get("/api/v1/conversations", response_model=list[Conversation])
    def list_conversations() -> tuple[Conversation, ...]:
        return store.list()

    @app.post("/api/v1/conversations", response_model=Conversation, status_code=201)
    def create_conversation(
        payload: ConversationCreate | None = None,
    ) -> Conversation:
        return store.create(payload.title if payload else None)

    @app.get("/api/v1/conversations/{conversation_id}", response_model=Conversation)
    def get_conversation(conversation_id: str) -> Conversation:
        conversation = store.get(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return conversation

    @app.patch("/api/v1/conversations/{conversation_id}", response_model=Conversation)
    def update_conversation(
        conversation_id: str, payload: ConversationUpdate
    ) -> Conversation:
        conversation = store.rename(conversation_id, payload.title)
        if conversation is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return conversation

    @app.delete("/api/v1/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict[str, Any]:
        if not store.delete(conversation_id):
            raise HTTPException(status_code=404, detail="conversation not found")
        return {}

    @app.get(
        "/api/v1/conversations/{conversation_id}/messages",
        response_model=list[Message],
    )
    def list_messages(conversation_id: str) -> tuple[Message, ...]:
        if store.get(conversation_id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return store.messages(conversation_id)

    @app.post("/api/v1/facts/read-source-window", response_model=SourceWindowResponse)
    def read_source_window(request: SourceWindowRequest) -> Any:
        reader = app.state.fact_reader
        try:
            if reader is None:
                reader = FactStore(settings.resolved_facts_root)
            evidence = reader.read_source_window(request.locator, request.window)
            return SourceWindowResponse(
                locator=request.locator,
                window=request.window,
                evidence=evidence,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return _api_error(
                409 if getattr(exc, "code", None) == "STALE_LOCATOR" else 400,
                str(getattr(exc, "code", "FACT_READ_ERROR")),
                str(exc),
            )

    @app.post("/api/v1/conversations/{conversation_id}/ask")
    def ask(conversation_id: str, request: AnswerRequest) -> StreamingResponse:
        if store.get(conversation_id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")

        async def events() -> AsyncIterator[bytes]:
            store.add_message(
                conversation_id, "user", request.question, mode=request.mode
            )
            history = store.messages(conversation_id)
            service = app.state.answer_service
            if service is None:
                yield _sse(
                    SseErrorEvent(
                        error=ApiError(
                            code="ANSWER_SERVICE_UNAVAILABLE",
                            message="No answer service is configured",
                            retryable=False,
                        )
                    )
                )
                return

            completed = False
            process_phases: list[AnswerProcessPhase] = []
            provider_counts: dict[str, int] = {}
            fused_count: int | None = None
            evidence_count = 0
            try:
                async for raw_event in service.stream_answer(
                    conversation_id=conversation_id,
                    request=request,
                    history=history,
                ):
                    event = _SSE_ADAPTER.validate_python(raw_event)
                    if isinstance(event, SsePhaseEvent):
                        process_phases.append(
                            AnswerProcessPhase(phase=event.phase, message=event.message)
                        )
                    elif isinstance(event, SseRetrievalSummaryEvent):
                        provider_counts = dict(event.provider_counts)
                        fused_count = event.fused_count
                    elif isinstance(event, SseEvidenceEvent):
                        evidence_count += 1
                    if isinstance(event, SseDoneEvent):
                        if completed:
                            raise ValueError("answer service emitted more than one done event")
                        completed = True
                        citations = tuple(
                            item.model_dump(mode="json")
                            for item in event.response.evidence
                        )
                        store.add_message(
                            conversation_id,
                            "assistant",
                            event.response.answer,
                            citations,
                            request.mode,
                            AnswerProcessTrace(
                                phases=tuple(process_phases),
                                provider_counts=provider_counts,
                                fused_count=fused_count,
                                evidence_count=max(evidence_count, len(citations)),
                            ),
                        )
                    yield _sse(event)
                if not completed:
                    yield _sse(
                        SseErrorEvent(
                            error=ApiError(
                                code="ANSWER_STREAM_INCOMPLETE",
                                message="Answer service ended without a done event",
                                retryable=True,
                            )
                        )
                    )
            # This is the API isolation boundary: arbitrary injected providers
            # are converted to the one allowed SSE error contract.
            except Exception as exc:  # noqa: BLE001
                yield _sse(SseErrorEvent(error=_error_from_exception(exc)))

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    _install_frontend(app, settings.resolved_frontend_dist_path)
    return app


def _build_default_answer_service(settings: Settings) -> AnswerService | None:
    """Construct the real direct service only when its mandatory inputs exist."""

    if not settings.deepseek_api_key or not settings.resolved_facts_root.is_dir():
        return None
    try:
        from .agentic import AgenticAnswerService, AnswerRouter
        from .direct import DirectAnswerService, LocalRetrieval
        from .providers import (
            DeepSeekClient,
            ImageObservationCache,
            ImageObserver,
            ZhipuEmbeddingClient,
            ZhipuVisionClient,
        )

        deepseek = DeepSeekClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            rewrite_model=settings.deepseek_rewrite_model,
            answer_model=settings.deepseek_answer_model,
            rewrite_prompt=settings.resolved_query_rewrite_prompt_path.read_text(
                encoding="utf-8"
            ),
            answer_prompt=settings.resolved_answer_prompt_path.read_text(
                encoding="utf-8"
            ),
            timeout=settings.provider_timeout_seconds,
        )
        embedder = None
        image_observer = None
        if settings.zhipu_api_key:
            embedder = ZhipuEmbeddingClient(
                api_key=settings.zhipu_api_key,
                base_url=settings.zhipu_base_url,
                model=settings.zhipu_embedding_model,
                dimensions=settings.zhipu_embedding_dimensions,
                timeout=settings.provider_timeout_seconds,
            )
            vision = ZhipuVisionClient(
                api_key=settings.zhipu_api_key,
                base_url=settings.zhipu_base_url,
                model=settings.zhipu_vision_model,
                prompt=settings.resolved_vision_prompt_path.read_text(encoding="utf-8"),
                timeout=settings.provider_timeout_seconds,
            )
            image_observer = ImageObserver(
                vision, ImageObservationCache(settings.resolved_chat_db_path)
            )
        retrieval = LocalRetrieval(
            lexical_path=settings.resolved_lexical_db_path,
            semantic_path=settings.resolved_semantic_db_path,
            embedder=embedder,
        )
        facts = FactStore(settings.resolved_facts_root)
        direct = DirectAnswerService(
            deepseek=deepseek,
            retrieval=retrieval,
            facts=facts,
            image_observer=image_observer,
        )
        agentic = AgenticAnswerService(
            deepseek=deepseek,
            retrieval=retrieval,
            facts=facts,
            system_prompt=settings.resolved_agentic_prompt_path.read_text(
                encoding="utf-8"
            ),
            image_observer=image_observer,
        )
        return AnswerRouter(
            direct=direct,
            agentic=agentic,
        )
    except (OSError, ValueError):
        return None


def _build_llm_wiki_adapter(settings: Settings) -> Any:
    from .llm_wiki import LlmWikiAdapter

    return LlmWikiAdapter(
        enabled=settings.llm_wiki_enabled,
        base_url=settings.llm_wiki_api_base_url,
        token=settings.llm_wiki_api_token,
        project_id=settings.llm_wiki_project_id,
        allow_remote=settings.llm_wiki_allow_remote,
        timeout=settings.provider_timeout_seconds,
    )


def _install_frontend(app: FastAPI, dist: Path) -> None:
    if not dist.is_dir() or not (dist / "index.html").is_file():
        return
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    def frontend_root() -> FileResponse:
        return FileResponse(dist / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend_route(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not found")
        candidate = (dist / path).resolve(strict=False)
        try:
            candidate.relative_to(dist.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="not found") from None
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


app = create_app()


def main() -> None:
    settings = Settings()
    uvicorn.run(
        "lenny_backend.app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
