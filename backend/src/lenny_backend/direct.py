from __future__ import annotations

import asyncio
import sqlite3
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ai_pm_contracts import (
    AnswerRequest,
    AnswerResponse,
    DocumentChunkLocator,
    Evidence,
    QueryRewrite,
    RetrievalHit,
    SseAnswerDeltaEvent,
    SseDoneEvent,
    SseEvent,
    SseEvidenceEvent,
    SsePhaseEvent,
    SseRetrievalSummaryEvent,
)
from retrieval_build import (
    connect_lexical,
    connect_semantic,
    get_index_status,
    keyword_search,
    reciprocal_rank_fusion,
    vector_search,
)

from .conversations import Message
from .providers import (
    DeepSeekClient,
    ImageObserver,
    ProviderError,
    fallback_query_rewrite,
    mark_images_uninspected,
)


class EmbeddingClient(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class FactReader(Protocol):
    def read_source_window(
        self, locator: DocumentChunkLocator, window: int = 0
    ) -> tuple[Evidence, ...]: ...


@dataclass(frozen=True)
class SearchOutcome:
    keyword_hits: tuple[RetrievalHit, ...]
    vector_hits: tuple[RetrievalHit, ...]
    errors: tuple[str, ...] = ()


class LocalRetrieval:
    """Independent lexical/vector readers; either side may degrade safely."""

    def __init__(
        self,
        *,
        lexical_path: str | Path,
        semantic_path: str | Path,
        embedder: EmbeddingClient | None,
    ):
        self.lexical_path = Path(lexical_path)
        self.semantic_path = Path(semantic_path)
        self.embedder = embedder

    async def search(self, rewrite: QueryRewrite) -> SearchOutcome:
        lexical_job = asyncio.to_thread(self._keyword, rewrite, 50)
        semantic_job = self.semantic(rewrite.semantic_query_en, limit=50)
        keyword_result, vector_result = await asyncio.gather(
            lexical_job, semantic_job, return_exceptions=True
        )
        errors: list[str] = []
        if isinstance(keyword_result, BaseException):
            errors.append(str(getattr(keyword_result, "code", "KEYWORD_SEARCH_ERROR")))
            keyword_hits: tuple[RetrievalHit, ...] = ()
        else:
            keyword_hits = keyword_result
        if isinstance(vector_result, BaseException):
            errors.append(str(getattr(vector_result, "code", "VECTOR_SEARCH_ERROR")))
            vector_hits: tuple[RetrievalHit, ...] = ()
        else:
            vector_hits = vector_result
        return SearchOutcome(keyword_hits, vector_hits, tuple(errors))

    async def keyword(self, query: str, *, limit: int = 50) -> tuple[RetrievalHit, ...]:
        _validate_tool_limit(limit)
        rewrite = fallback_query_rewrite(query)
        return await asyncio.to_thread(self._keyword, rewrite, limit)

    def _keyword(self, rewrite: QueryRewrite, limit: int) -> tuple[RetrievalHit, ...]:
        if not self.lexical_path.is_file():
            raise ProviderError("LEXICAL_INDEX_NOT_READY", "lexical index does not exist")
        with connect_lexical(self.lexical_path, query_only=True) as connection:
            return keyword_search(connection, rewrite, limit=limit)

    async def semantic(self, query: str, *, limit: int = 50) -> tuple[RetrievalHit, ...]:
        _validate_tool_limit(limit)
        if self.embedder is None:
            raise ProviderError(
                "SEMANTIC_PROVIDER_UNAVAILABLE", "Zhipu embedding is not configured"
            )
        if not self.semantic_path.is_file():
            raise ProviderError("SEMANTIC_INDEX_NOT_READY", "semantic index does not exist")
        embedding = await self.embedder.embed(query)
        return await asyncio.to_thread(self._vector, embedding, limit)

    def _vector(
        self, embedding: Sequence[float], limit: int
    ) -> tuple[RetrievalHit, ...]:
        with connect_semantic(self.semantic_path, query_only=True) as connection:
            status = get_index_status(connection)
            if status.status != "ready":
                raise ProviderError(
                    "SEMANTIC_INDEX_NOT_PRODUCTION_READY",
                    f"semantic index status is {status.status}",
                )
            return vector_search(connection, embedding, limit=limit)


class RetrievalService(Protocol):
    async def search(self, rewrite: QueryRewrite) -> SearchOutcome: ...


class DirectAnswerService:
    def __init__(
        self,
        *,
        deepseek: DeepSeekClient,
        retrieval: RetrievalService,
        facts: FactReader,
        image_observer: ImageObserver | None = None,
    ):
        self.deepseek = deepseek
        self.retrieval = retrieval
        self.facts = facts
        self.image_observer = image_observer

    async def stream_answer(
        self,
        *,
        conversation_id: str,
        request: AnswerRequest,
        history: Sequence[Message],
    ) -> AsyncIterator[SseEvent]:
        if request.mode != "direct":
            raise ProviderError(
                "AGENTIC_RAG_DISABLED",
                "The direct answer service cannot execute agentic mode",
            )
        context = build_recent_context(history, current_question=request.question)

        yield SsePhaseEvent(phase="rewrite", message="Preparing retrieval query")
        rewrite = await self.deepseek.rewrite(request.question, context)

        yield SsePhaseEvent(phase="retrieve", message="Searching local indexes")
        outcome = await self.retrieval.search(rewrite)
        fused = reciprocal_rank_fusion(
            (outcome.keyword_hits, outcome.vector_hits), k=60, limit=12
        )
        yield SseRetrievalSummaryEvent(
            provider_counts={
                "keyword": len(outcome.keyword_hits),
                "vector": len(outcome.vector_hits),
            },
            fused_count=len(fused),
        )

        yield SsePhaseEvent(phase="resolve", message="Reading fact-layer evidence")
        resolved: list[Evidence] = []
        seen: set[str] = set()
        for hit in fused:
            if not isinstance(hit.locator, DocumentChunkLocator):
                continue
            try:
                # Direct retrieval returns the fused parent hit plus one
                # neighboring HybridChunker block on either side. A single
                # block may itself resolve to many Docling item references.
                window = self.facts.read_source_window(hit.locator, 1)
            except (OSError, RuntimeError, ValueError, sqlite3.Error):
                # A stale/bad locator is not evidence and is skipped fail-closed.
                continue
            for evidence in window:
                if evidence.evidence_id not in seen:
                    seen.add(evidence.evidence_id)
                    resolved.append(evidence)

        evidence_items = tuple(resolved)
        if evidence_items:
            if self.image_observer is None:
                fallback = mark_images_uninspected(
                    evidence_items, error="ZHIPU_VISION_NOT_CONFIGURED"
                )
                if fallback != evidence_items:
                    yield SsePhaseEvent(
                        phase="inspect_images",
                        message="Vision unavailable; preserving image references",
                    )
                evidence_items = fallback
            else:
                yield SsePhaseEvent(
                    phase="inspect_images", message="Inspecting referenced images"
                )
                try:
                    evidence_items = await self.image_observer.enrich(evidence_items)
                except (OSError, RuntimeError, ValueError, sqlite3.Error):
                    # Preserve URL/caption and an explicit failure marker for the answer.
                    evidence_items = mark_images_uninspected(
                        evidence_items, error="ZHIPU_VISION_UNAVAILABLE"
                    )

        for evidence in evidence_items:
            yield SseEvidenceEvent(evidence=evidence)

        yield SsePhaseEvent(phase="answer", message="Answering from resolved facts")
        answer_parts: list[str] = []
        async for delta in self.deepseek.stream_answer(
            question=request.question,
            conversation_context=context,
            evidence=evidence_items,
        ):
            if delta:
                answer_parts.append(delta)
                yield SseAnswerDeltaEvent(delta=delta)
        answer = "".join(answer_parts).strip()
        if not answer:
            raise ProviderError(
                "EMPTY_DEEPSEEK_ANSWER", "DeepSeek returned no answer text", retryable=True
            )
        yield SseDoneEvent(
            response=AnswerResponse(
                request_id=str(uuid.uuid4()),
                mode="direct",
                answer=answer,
                evidence=evidence_items,
            )
        )


def build_recent_context(
    history: Sequence[Message],
    *,
    current_question: str,
    max_rounds: int = 12,
    max_tokens: int = 24_000,
) -> str:
    """Return newest context, bounded by rounds and a conservative token estimate."""

    messages = list(history)
    if (
        messages
        and messages[-1].role == "user"
        and messages[-1].content == current_question
    ):
        messages.pop()
    messages = messages[-(max_rounds * 2) :]
    selected: list[str] = []
    used = 0
    for message in reversed(messages):
        line = f"{message.role}: {message.content}"
        separator_tokens = 1 if selected else 0
        remaining = max_tokens - used - separator_tokens
        if remaining <= 0:
            break
        estimated = conservative_token_estimate(line)
        if estimated > remaining:
            line = _take_last_utf8_bytes(line, remaining)
            estimated = conservative_token_estimate(line)
        selected.append(line)
        used += estimated + separator_tokens
    return "\n".join(reversed(selected))


def conservative_token_estimate(text: str) -> int:
    """UTF-8 bytes upper-bound ordinary byte-level BPE tokens for plain text."""

    return len(text.encode("utf-8"))


def _take_last_utf8_bytes(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    result: list[str] = []
    used = 0
    for character in reversed(text):
        size = len(character.encode("utf-8"))
        if used + size > budget:
            break
        result.append(character)
        used += size
    return "".join(reversed(result))


def _validate_tool_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ProviderError("INVALID_SEARCH_LIMIT", "search limit must be between 1 and 50")
