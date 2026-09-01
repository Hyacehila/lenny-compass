from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from ai_pm_contracts import (
    AnswerClaim,
    AnswerProcessPhase,
    AnswerProcessTrace,
    AnswerRequest,
    AnswerResponse,
    ApiError,
    ApiErrorResponse,
    Evidence,
    SourceWindowRequest,
    SourceWindowResponse,
    SseAnswerDeltaEvent,
    SseDoneEvent,
    SseErrorEvent,
    SseEvent,
    SseEvidenceEvent,
    SsePhaseEvent,
    SseRetrievalSummaryEvent,
)
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from retrieval_build import FactStore
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .conversations import (
    AnswerRun,
    Conversation,
    ConversationBusyError,
    ConversationNotFoundError,
    ConversationStore,
    IdempotencyConflictError,
    Message,
)
from .grounding import insufficient_evidence_answer
from .providers import ProviderError
from .settings import Settings
from .snapshot import RetrievalSnapshot, SnapshotFactStore

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


def _answer_stream(events: AsyncIterator[bytes], run_id: str) -> StreamingResponse:
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Answer-Run-ID": run_id,
        },
    )


async def _fail_run_safely(
    store: ConversationStore,
    run_id: str,
    *,
    code: str,
    message: str,
    retryable: bool,
    cancelled: bool = False,
) -> None:
    try:
        await asyncio.to_thread(
            store.fail_run,
            run_id,
            code=code,
            message=message,
            retryable=retryable,
            cancelled=cancelled,
        )
    except Exception:  # noqa: BLE001
        # The SSE terminal must survive a secondary persistence failure.
        return


async def _close_answer_stream(stream: Any) -> None:
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        await asyncio.wait_for(close(), timeout=1.0)
    except (Exception, asyncio.CancelledError):  # noqa: BLE001
        return


def _error_from_exception(exc: Exception) -> ApiError:
    if isinstance(exc, ProviderError):
        return ApiError(
            code=exc.code,
            message=str(exc) or "Answer provider failed",
            retryable=exc.retryable,
        )
    if isinstance(exc, (TypeError, ValueError, ValidationError)):
        return ApiError(
            code="ANSWER_SERVICE_CONTRACT_ERROR",
            message="Answer service returned an invalid response",
            retryable=False,
        )
    return ApiError(
        code="ANSWER_SERVICE_ERROR",
        message="Answer service failed before completion",
        retryable=True,
    )


def _read_health(
    settings: Settings,
    answer_service: AnswerService | None = None,
    startup_error: str | None = None,
    retrieval_snapshot: RetrievalSnapshot | None = None,
    retrieval_snapshot_error: str | None = None,
) -> dict[str, Any]:
    facts_ready = False
    lexical_ready = False
    semantic_ready = False
    document_count = 0
    chunk_count = 0
    detail: dict[str, Any] = {}

    snapshot = retrieval_snapshot or getattr(answer_service, "retrieval_snapshot", None)
    if isinstance(snapshot, RetrievalSnapshot):
        try:
            snapshot.assert_unchanged()
            manifest = snapshot.facts_manifest
            bundle = snapshot.bundle.manifest
            facts_ready = True
            document_count = manifest.document_count
            chunk_count = manifest.chunk_count
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            detail["retrieval"] = (
                f"{getattr(exc, 'code', 'RETRIEVAL_SNAPSHOT_CHANGED')}: "
                "validated retrieval snapshot is no longer available"
            )
            bundle = None
    else:
        bundle = None
        detail["retrieval"] = retrieval_snapshot_error or (
            "retrieval snapshot was not validated at startup"
        )

    if bundle is not None:
        lexical_ready = bundle.lexical_status == "ready"
        # test_only artifacts are intentionally not production-ready.
        semantic_ready = bundle.semantic_status == "ready"
        if bundle.semantic_reason:
            detail["semantic_reason"] = bundle.semantic_reason
        detail["retrieval_build_id"] = bundle.build_id
    if startup_error:
        detail["answer_service"] = startup_error

    deepseek_ready = bool(settings.deepseek_api_key)
    claim_verifier_ready = deepseek_ready and all(
        path.is_file()
        for path in (
            settings.resolved_question_plan_prompt_path,
            settings.resolved_stage2_answer_prompt_path,
            settings.resolved_semantic_verify_prompt_path,
            settings.resolved_stage2_repair_prompt_path,
        )
    )
    zhipu_ready = bool(settings.zhipu_api_key)
    local_ready = facts_ready and lexical_ready
    answer_ready = local_ready and deepseek_ready and answer_service is not None
    full_ready = answer_ready and semantic_ready and zhipu_ready
    return {
        "status": "ready"
        if full_ready
        else "answer_ready"
        if answer_ready
        else "degraded",
        "agentic_enabled": bool(
            answer_service is not None
            and getattr(answer_service, "agentic_available", False)
        ),
        "facts_ready": facts_ready,
        "lexical_ready": lexical_ready,
        "semantic_ready": semantic_ready,
        "deepseek_ready": deepseek_ready,
        "claim_verifier_ready": claim_verifier_ready,
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
    retrieval_snapshot: RetrievalSnapshot | None = None
    retrieval_snapshot_error: str | None = None
    try:
        retrieval_snapshot = RetrievalSnapshot.load(
            settings.resolved_retrieval_manifest_path,
            facts_root=settings.resolved_facts_root,
        )
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        retrieval_snapshot_error = (
            f"{getattr(exc, 'code', type(exc).__name__)}: "
            "retrieval snapshot validation failed at startup"
        )
    answer_service_startup_error: str | None = None
    if answer_service is None:
        if (
            settings.deepseek_api_key
            and settings.resolved_facts_root.is_dir()
            and retrieval_snapshot is None
        ):
            answer_service_startup_error = retrieval_snapshot_error
        else:
            try:
                answer_service = _build_default_answer_service(
                    settings, snapshot=retrieval_snapshot
                )
            except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
                answer_service_startup_error = (
                    f"{getattr(exc, 'code', type(exc).__name__)}: {exc}"
                )
                answer_service = None
    app = FastAPI(title="Lenny API", version="0.1.0")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(trusted_hosts or ("localhost", "127.0.0.1")),
    )
    app.state.settings = settings
    app.state.conversation_store = store
    app.state.answer_service = answer_service
    app.state.answer_service_startup_error = answer_service_startup_error
    app.state.retrieval_snapshot = retrieval_snapshot or getattr(
        answer_service, "retrieval_snapshot", None
    )
    app.state.retrieval_snapshot_error = retrieval_snapshot_error
    app.state.fact_reader = (
        fact_reader
        or getattr(answer_service, "fact_reader", None)
        or (
            SnapshotFactStore(app.state.retrieval_snapshot)
            if isinstance(app.state.retrieval_snapshot, RetrievalSnapshot)
            else None
        )
    )
    app.state.llm_wiki_adapter = _build_llm_wiki_adapter(settings)

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return _read_health(
            settings,
            app.state.answer_service,
            app.state.answer_service_startup_error,
            app.state.retrieval_snapshot,
            app.state.retrieval_snapshot_error,
        )

    @app.get("/api/v1/index/status")
    def index_status() -> Any:
        try:
            snapshot = app.state.retrieval_snapshot
            if isinstance(snapshot, RetrievalSnapshot):
                snapshot.assert_unchanged()
                bundle = snapshot.bundle.manifest
            else:
                return _api_error(
                    503,
                    "INDEX_NOT_READY",
                    app.state.retrieval_snapshot_error
                    or "retrieval snapshot was not validated at startup",
                    retryable=True,
                )
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            return _api_error(
                503,
                str(getattr(exc, "code", "RETRIEVAL_SNAPSHOT_CHANGED")),
                "The validated retrieval snapshot is no longer available",
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
    def delete_conversation(conversation_id: str) -> Any:
        try:
            deleted = store.delete(conversation_id)
        except ConversationBusyError as exc:
            return _api_error(
                409,
                "ANSWER_RUN_IN_PROGRESS",
                "Stop the active answer run before deleting this conversation",
                retryable=True,
                details={"run_id": exc.active_run.id},
            )
        if not deleted:
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

    @app.get(
        "/api/v1/conversations/{conversation_id}/runs",
        response_model=list[AnswerRun],
    )
    def list_answer_runs(conversation_id: str) -> tuple[AnswerRun, ...]:
        if store.get(conversation_id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return store.runs(conversation_id)

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
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            return _api_error(
                409 if getattr(exc, "code", None) == "STALE_LOCATOR" else 400,
                str(getattr(exc, "code", "FACT_READ_ERROR")),
                str(exc),
            )

    @app.post("/api/v1/conversations/{conversation_id}/ask")
    def ask(conversation_id: str, request: AnswerRequest) -> Any:
        try:
            run, created = store.start_run(
                conversation_id,
                request.question,
                request.mode,
                idempotency_key=request.idempotency_key,
            )
        except ConversationNotFoundError:
            raise HTTPException(
                status_code=404, detail="conversation not found"
            ) from None
        except ConversationBusyError as exc:
            return _api_error(
                409,
                "ANSWER_RUN_IN_PROGRESS",
                "This conversation already has an active answer run",
                retryable=True,
                details={"run_id": exc.active_run.id},
            )
        except IdempotencyConflictError as exc:
            return _api_error(
                409,
                "IDEMPOTENCY_KEY_CONFLICT",
                str(exc),
                details={"idempotency_key": request.idempotency_key or ""},
            )

        if not created:

            async def replay() -> AsyncIterator[bytes]:
                if run.status == "succeeded" and run.assistant_message_id:
                    try:
                        message = await asyncio.to_thread(
                            store.message, run.assistant_message_id
                        )
                        if message is None or (
                            message.contract_version == "v1"
                            and message.answer_kind is None
                        ):
                            raise ValueError(
                                "completed answer run has no persisted response"
                            )
                        response = AnswerResponse.model_validate(
                            {
                                "request_id": run.id,
                                "assistant_message_id": message.id,
                                "mode": run.mode,
                                "contract_version": message.contract_version,
                                "response_kind": message.response_kind,
                                "answer_kind": message.answer_kind,
                                "answer": message.content,
                                "evidence": tuple(
                                    Evidence.model_validate(item)
                                    for item in message.citations
                                ),
                                "claims": tuple(
                                    AnswerClaim.model_validate(item)
                                    for item in message.claims
                                ),
                                "advice": message.response_payload.get("advice", ()),
                                "user_premises": message.response_payload.get(
                                    "user_premises", ()
                                ),
                                "clarification": message.response_payload.get(
                                    "clarification"
                                ),
                                "verification": message.response_payload.get(
                                    "verification"
                                ),
                                "insufficient_reason": message.response_payload.get(
                                    "insufficient_reason"
                                ),
                            }
                        )
                    except Exception:  # noqa: BLE001
                        error = ApiError(
                            code="ANSWER_RUN_CORRUPT",
                            message="Completed answer run has no persisted response",
                            retryable=True,
                            details={"run_id": run.id},
                        )
                        yield _sse(SseErrorEvent(error=error))
                        return
                    yield _sse(SseDoneEvent(response=response))
                    return
                error = ApiError(
                    code=run.error_code
                    or (
                        "ANSWER_RUN_IN_PROGRESS"
                        if run.status in {"queued", "running"}
                        else "ANSWER_RUN_CANCELLED"
                    ),
                    message=run.error_message
                    or (
                        "The identical answer run is still in progress"
                        if run.status in {"queued", "running"}
                        else "The identical answer run was cancelled"
                    ),
                    retryable=run.retryable or run.status != "succeeded",
                    details={"run_id": run.id},
                )
                yield _sse(SseErrorEvent(error=error))

            return _answer_stream(replay(), run.id)

        async def events() -> AsyncIterator[bytes]:
            service = app.state.answer_service
            done_event: SseDoneEvent | None = None
            service_error: ApiError | None = None
            answer_deltas: list[SseAnswerDeltaEvent] = []
            process_phases: list[AnswerProcessPhase] = []
            provider_counts: dict[str, int] = {}
            fused_count: int | None = None
            event_count = 0
            evidence_event_count = 0
            answer_delta_bytes = 0
            try:
                if service is None:
                    error = ApiError(
                        code="ANSWER_SERVICE_UNAVAILABLE",
                        message="No answer service is configured",
                        retryable=False,
                        details={"run_id": run.id},
                    )
                    await _fail_run_safely(
                        store,
                        run.id,
                        code=error.code,
                        message=error.message,
                        retryable=error.retryable,
                    )
                    yield _sse(SseErrorEvent(error=error))
                    return

                history = await asyncio.to_thread(store.messages, conversation_id)
                stream = service.stream_answer(
                    conversation_id=conversation_id,
                    request=request,
                    history=history,
                )
                try:
                    async with asyncio.timeout(settings.answer_run_timeout_seconds):
                        async for raw_event in stream:
                            event_count += 1
                            if event_count > 2_048:
                                raise ValueError(
                                    "answer service exceeded the SSE event limit"
                                )
                            event = _SSE_ADAPTER.validate_python(raw_event)
                            if isinstance(event, SsePhaseEvent):
                                if len(process_phases) >= 128:
                                    raise ValueError(
                                        "answer service exceeded the phase event limit"
                                    )
                                process_phases.append(
                                    AnswerProcessPhase(
                                        phase=event.phase, message=event.message
                                    )
                                )
                            elif isinstance(event, SseRetrievalSummaryEvent):
                                provider_counts = dict(event.provider_counts)
                                fused_count = event.fused_count
                            elif isinstance(event, SseEvidenceEvent):
                                evidence_event_count += 1
                                if evidence_event_count > 128:
                                    raise ValueError(
                                        "answer service exceeded the evidence event limit"
                                    )

                            if isinstance(event, SseDoneEvent):
                                done_event = event
                                break
                            if isinstance(event, SseErrorEvent):
                                service_error = event.error
                                break
                            if isinstance(event, SseAnswerDeltaEvent):
                                answer_delta_bytes += len(event.delta.encode("utf-8"))
                                if answer_delta_bytes > 100_000:
                                    raise ValueError(
                                        "answer service exceeded the answer byte limit"
                                    )
                                answer_deltas.append(event)
                            else:
                                yield _sse(event)
                finally:
                    await _close_answer_stream(stream)

                if service_error is not None:
                    error = service_error.model_copy(
                        update={
                            "details": {
                                **service_error.details,
                                "run_id": run.id,
                            }
                        }
                    )
                    await _fail_run_safely(
                        store,
                        run.id,
                        code=error.code,
                        message=error.message,
                        retryable=error.retryable,
                    )
                    yield _sse(SseErrorEvent(error=error))
                    return
                if done_event is None:
                    error = ApiError(
                        code="ANSWER_STREAM_INCOMPLETE",
                        message="Answer service ended without a done event",
                        retryable=True,
                        details={"run_id": run.id},
                    )
                    await _fail_run_safely(
                        store,
                        run.id,
                        code=error.code,
                        message=error.message,
                        retryable=error.retryable,
                    )
                    yield _sse(SseErrorEvent(error=error))
                    return

                validated_response = AnswerResponse.model_validate(
                    done_event.response.model_dump(mode="python")
                )
                if validated_response.mode != run.mode:
                    raise ValueError(
                        "answer service response mode does not match the answer run"
                    )
                if (
                    validated_response.answer_kind == "insufficient_evidence"
                    and validated_response.answer
                    != insufficient_evidence_answer(request.question)
                ):
                    raise ValueError(
                        "insufficient_evidence response is not the deterministic refusal"
                    )
                if answer_deltas and (
                    "".join(event.delta for event in answer_deltas).strip()
                    != validated_response.answer
                ):
                    raise ValueError("answer deltas do not match the terminal response")
                response = validated_response.model_copy(update={"request_id": run.id})
                citations = tuple(
                    item.model_dump(mode="json") for item in response.evidence
                )
                claims = tuple(item.model_dump(mode="json") for item in response.claims)
                response_payload = {
                    "advice": [
                        item.model_dump(mode="json") for item in response.advice
                    ],
                    "user_premises": [
                        item.model_dump(mode="json")
                        for item in response.user_premises
                    ],
                    "clarification": (
                        response.clarification.model_dump(mode="json")
                        if response.clarification
                        else None
                    ),
                    "verification": (
                        response.verification.model_dump(mode="json")
                        if response.verification
                        else None
                    ),
                    "insufficient_reason": response.insufficient_reason,
                }
                trace = AnswerProcessTrace(
                    phases=tuple(process_phases),
                    provider_counts=provider_counts,
                    fused_count=fused_count,
                    evidence_count=len(citations),
                )
                _, assistant_message = await asyncio.to_thread(
                    store.complete_run,
                    run.id,
                    answer=response.answer,
                    citations=citations,
                    claims=claims,
                    answer_kind=response.answer_kind,
                    process_trace=trace,
                    contract_version=response.contract_version,
                    response_kind=response.response_kind,
                    response_payload=response_payload,
                )
                response = response.model_copy(
                    update={"assistant_message_id": assistant_message.id}
                )
                for delta_event in answer_deltas:
                    yield _sse(delta_event)
                yield _sse(SseDoneEvent(response=response))
            except asyncio.CancelledError:
                await _fail_run_safely(
                    store,
                    run.id,
                    code="ANSWER_RUN_CANCELLED",
                    message="Answer stream was cancelled by the client",
                    retryable=True,
                    cancelled=True,
                )
                raise
            except TimeoutError:
                error = ApiError(
                    code="ANSWER_RUN_TIMEOUT",
                    message="Answer run exceeded its total deadline",
                    retryable=True,
                    details={"run_id": run.id},
                )
                await _fail_run_safely(
                    store,
                    run.id,
                    code=error.code,
                    message=error.message,
                    retryable=error.retryable,
                )
                yield _sse(SseErrorEvent(error=error))
            except Exception as exc:  # noqa: BLE001
                raw_error = _error_from_exception(exc)
                error = raw_error.model_copy(
                    update={"details": {**raw_error.details, "run_id": run.id}}
                )
                await _fail_run_safely(
                    store,
                    run.id,
                    code=error.code,
                    message=error.message,
                    retryable=error.retryable,
                )
                yield _sse(SseErrorEvent(error=error))

        return _answer_stream(events(), run.id)

    _install_frontend(app, settings.resolved_frontend_dist_path)
    return app


def _build_default_answer_service(
    settings: Settings,
    *,
    snapshot: RetrievalSnapshot | None = None,
) -> AnswerService | None:
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

        if snapshot is None:
            raise RuntimeError("retrieval snapshot was not validated at startup")
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
            plan_prompt=settings.resolved_question_plan_prompt_path.read_text(
                encoding="utf-8"
            ),
            stage2_answer_prompt=settings.resolved_stage2_answer_prompt_path.read_text(
                encoding="utf-8"
            ),
            verify_prompt=settings.resolved_semantic_verify_prompt_path.read_text(
                encoding="utf-8"
            ),
            repair_prompt=settings.resolved_stage2_repair_prompt_path.read_text(
                encoding="utf-8"
            ),
            verify_model=settings.deepseek_verify_model,
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
            lexical_path=snapshot.lexical_path,
            semantic_path=snapshot.semantic_path,
            embedder=embedder,
            snapshot=snapshot,
        )
        facts = SnapshotFactStore(snapshot)
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
            timeout_seconds=settings.answer_run_timeout_seconds,
        )
        router = AnswerRouter(
            direct=direct,
            agentic=agentic,
        )
        router.retrieval_snapshot = snapshot
        router.fact_reader = facts
        return router
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"answer runtime snapshot initialization failed: {exc}"
        ) from exc


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
