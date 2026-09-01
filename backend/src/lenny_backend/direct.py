from __future__ import annotations

import asyncio
import sqlite3
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

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
from .grounding import bind_cited_evidence, insufficient_evidence_answer
from .providers import (
    DeepSeekClient,
    ImageObserver,
    ProviderError,
    fallback_query_rewrite,
    mark_images_uninspected,
)
from .stage2 import (
    coaching_state,
    finalize_stage2_answer,
    make_clarification_response,
    supports_stage2,
)

if TYPE_CHECKING:
    from .snapshot import RetrievalSnapshot


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
        semantic_path: str | Path | None,
        embedder: EmbeddingClient | None,
        snapshot: RetrievalSnapshot | None = None,
    ):
        self.lexical_path = Path(lexical_path)
        self.semantic_path = Path(semantic_path) if semantic_path is not None else None
        self.embedder = embedder
        self.snapshot = snapshot

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
        if self.snapshot is not None:
            self.snapshot.assert_unchanged()
        if not self.lexical_path.is_file():
            raise ProviderError(
                "LEXICAL_INDEX_NOT_READY", "lexical index does not exist"
            )
        with closing(connect_lexical(self.lexical_path, query_only=True)) as connection:
            return keyword_search(connection, rewrite, limit=limit)

    async def semantic(
        self, query: str, *, limit: int = 50
    ) -> tuple[RetrievalHit, ...]:
        _validate_tool_limit(limit)
        if self.embedder is None:
            raise ProviderError(
                "SEMANTIC_PROVIDER_UNAVAILABLE", "Zhipu embedding is not configured"
            )
        if self.semantic_path is None or not self.semantic_path.is_file():
            raise ProviderError(
                "SEMANTIC_INDEX_NOT_READY", "semantic index does not exist"
            )
        embedding = await self.embedder.embed(query)
        return await asyncio.to_thread(self._vector, embedding, limit)

    def _vector(
        self, embedding: Sequence[float], limit: int
    ) -> tuple[RetrievalHit, ...]:
        if self.snapshot is not None:
            self.snapshot.assert_unchanged()
        if self.semantic_path is None:
            raise ProviderError(
                "SEMANTIC_INDEX_NOT_READY", "semantic index does not exist"
            )
        with closing(
            connect_semantic(self.semantic_path, query_only=True)
        ) as connection:
            status = get_index_status(connection)
            if status.status != "ready":
                raise ProviderError(
                    "SEMANTIC_INDEX_NOT_PRODUCTION_READY",
                    f"semantic index status is {status.status}",
                )
            return vector_search(connection, embedding, limit=limit)


class RetrievalService(Protocol):
    async def search(self, rewrite: QueryRewrite) -> SearchOutcome: ...


def assert_runtime_snapshots_unchanged(*sources: Any) -> None:
    """Fail the whole run if any bound immutable generation changed."""

    checked: set[int] = set()
    for source in sources:
        snapshot = getattr(source, "snapshot", None)
        if snapshot is None or id(snapshot) in checked:
            continue
        checked.add(id(snapshot))
        try:
            snapshot.assert_unchanged()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProviderError(
                "RETRIEVAL_SNAPSHOT_CHANGED",
                "The validated retrieval snapshot changed during the answer run",
                retryable=True,
            ) from exc


class DirectAnswerService:
    def __init__(
        self,
        *,
        deepseek: DeepSeekClient,
        retrieval: RetrievalService,
        facts: FactReader,
        image_observer: ImageObserver | None = None,
        max_answer_tokens: int = 64_000,
    ):
        if max_answer_tokens <= 0:
            raise ValueError("max_answer_tokens must be positive")
        self.deepseek = deepseek
        self.retrieval = retrieval
        self.facts = facts
        self.image_observer = image_observer
        self.max_answer_tokens = max_answer_tokens

    async def stream_answer(
        self,
        *,
        conversation_id: str,
        request: AnswerRequest,
        history: Sequence[Message],
    ) -> AsyncIterator[SseEvent]:
        if request.mode != "direct":
            raise ProviderError(
                "INVALID_ANSWER_MODE",
                "The direct answer service cannot execute agentic mode",
            )
        context = build_recent_context(history, current_question=request.question)
        stage2_enabled = supports_stage2(self.deepseek)
        source_scope = "single_document"
        if stage2_enabled:
            completed_rounds, coaching_id = coaching_state(
                history,
                coaching_id=request.coaching_id,
                reply_to_message_id=request.reply_to_message_id,
            )
            yield SsePhaseEvent(phase="plan", message="正在判断问题和所需信息")
            plan = await self.deepseek.plan_question(
                question=request.question,
                conversation_context=context,
                completed_clarification_rounds=completed_rounds,
                skip_clarification=request.skip_clarification,
            )
            if (
                plan.needs_clarification
                and not request.skip_clarification
                and completed_rounds < 3
            ):
                response = make_clarification_response(
                    request_id=str(uuid.uuid4()),
                    mode="direct",
                    plan=plan,
                    completed_rounds=completed_rounds,
                    coaching_id=coaching_id,
                )
                yield SseAnswerDeltaEvent(delta=response.answer)
                yield SseDoneEvent(response=response)
                return
            rewrite = plan.as_rewrite(request.question)
            source_scope = plan.source_scope
        else:
            yield SsePhaseEvent(phase="rewrite", message="正在理解问题")
            rewrite = await self.deepseek.rewrite(request.question, context)

        yield SsePhaseEvent(phase="retrieve", message="正在检索本地知识库")
        outcome = await self.retrieval.search(rewrite)
        if "RETRIEVAL_SNAPSHOT_CHANGED" in outcome.errors:
            raise ProviderError(
                "RETRIEVAL_SNAPSHOT_CHANGED",
                "The validated retrieval snapshot changed during the answer run",
                retryable=True,
            )
        assert_runtime_snapshots_unchanged(self.retrieval, self.facts)
        if len(outcome.errors) >= 2:
            raise ProviderError(
                "RETRIEVAL_UNAVAILABLE",
                "All configured retrieval providers failed",
                retryable=True,
            )
        fused_candidates = reciprocal_rank_fusion(
            (outcome.keyword_hits, outcome.vector_hits),
            k=60,
            limit=30 if stage2_enabled and source_scope == "broad" else 12,
        )
        fused = (
            _diversify_hits(fused_candidates, limit=12, per_document=3)
            if stage2_enabled and source_scope == "broad"
            else fused_candidates
        )
        yield SseRetrievalSummaryEvent(
            provider_counts={
                "keyword": len(outcome.keyword_hits),
                "vector": len(outcome.vector_hits),
            },
            fused_count=len(fused),
        )

        yield SsePhaseEvent(phase="resolve", message="正在读取原文证据")
        resolved: list[Evidence] = []
        seen: set[str] = set()
        resolvable_hits = 0
        resolution_failures = 0
        for hit in fused:
            if not isinstance(hit.locator, DocumentChunkLocator):
                continue
            resolvable_hits += 1
            try:
                # Direct retrieval returns the fused parent hit plus one
                # neighboring HybridChunker block on either side. A single
                # block may itself resolve to many Docling item references.
                window = await asyncio.to_thread(
                    self.facts.read_source_window, hit.locator, 1
                )
            except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
                if getattr(exc, "code", None) == "RETRIEVAL_SNAPSHOT_CHANGED":
                    raise ProviderError(
                        "RETRIEVAL_SNAPSHOT_CHANGED",
                        "The validated retrieval snapshot changed during the answer run",
                        retryable=True,
                    ) from exc
                # A stale/bad locator is not evidence and is skipped fail-closed.
                resolution_failures += 1
                continue
            for evidence in window:
                if evidence.evidence_id not in seen:
                    seen.add(evidence.evidence_id)
                    resolved.append(evidence)

        evidence_items = tuple(resolved)
        if fused and (resolvable_hits == 0 or resolution_failures == resolvable_hits):
            raise ProviderError(
                "FACT_RESOLUTION_FAILED",
                "Retrieved candidates could not be resolved against the facts snapshot",
                retryable=True,
            )
        if evidence_items:
            if self.image_observer is None:
                fallback = mark_images_uninspected(
                    evidence_items, error="ZHIPU_VISION_NOT_CONFIGURED"
                )
                if fallback != evidence_items:
                    yield SsePhaseEvent(
                        phase="inspect_images",
                        message="视觉模型暂不可用，保留原始图片引用",
                    )
                evidence_items = fallback
            else:
                yield SsePhaseEvent(phase="inspect_images", message="正在分析相关图片")
                try:
                    evidence_items = await self.image_observer.enrich(evidence_items)
                except (OSError, RuntimeError, ValueError, sqlite3.Error):
                    # Preserve URL/caption and an explicit failure marker for the answer.
                    evidence_items = mark_images_uninspected(
                        evidence_items, error="ZHIPU_VISION_UNAVAILABLE"
                    )

        assert_runtime_snapshots_unchanged(self.retrieval, self.facts)

        if not stage2_enabled:
            for evidence in evidence_items:
                yield SseEvidenceEvent(evidence=evidence)

        if stage2_enabled:
            yield SsePhaseEvent(phase="answer", message="正在组织结论和建议")
            draft = await self.deepseek.draft_stage2_answer(
                question=request.question,
                conversation_context=context,
                evidence=evidence_items,
                message_ids=tuple(message.id for message in history),
            )
            yield SsePhaseEvent(
                phase="verify",
                message="正在独立核验每条结论，必要时自动修复一次",
            )
            response = await finalize_stage2_answer(
                provider=self.deepseek,
                mode="direct",
                request_id=str(uuid.uuid4()),
                question=request.question,
                conversation_context=context,
                evidence=evidence_items,
                history=history,
                initial_draft=draft,
            )
            assert_runtime_snapshots_unchanged(self.retrieval, self.facts)
            if conservative_token_estimate(response.answer) > self.max_answer_tokens:
                raise ProviderError(
                    "DIRECT_ANSWER_LIMIT",
                    f"Answer exceeds {self.max_answer_tokens} conservative tokens",
                )
            for evidence in response.evidence:
                yield SseEvidenceEvent(evidence=evidence)
            yield SseAnswerDeltaEvent(delta=response.answer)
            yield SseDoneEvent(response=response)
            return

        yield SsePhaseEvent(phase="answer", message="正在根据证据生成回答")
        if not evidence_items:
            answer = insufficient_evidence_answer(request.question)
            yield SseAnswerDeltaEvent(delta=answer)
            yield SseDoneEvent(
                response=AnswerResponse(
                    request_id=str(uuid.uuid4()),
                    mode="direct",
                    answer_kind="insufficient_evidence",
                    answer=answer,
                )
            )
            return

        answer_parts: list[str] = []
        answer_tokens = 0
        refusal = insufficient_evidence_answer(request.question)
        async for delta in self.deepseek.stream_answer(
            question=request.question,
            conversation_context=context,
            evidence=evidence_items,
            insufficient_answer=refusal,
        ):
            if delta:
                answer_tokens += conservative_token_estimate(delta)
                if answer_tokens > self.max_answer_tokens:
                    raise ProviderError(
                        "DIRECT_ANSWER_LIMIT",
                        f"Answer exceeds {self.max_answer_tokens} conservative tokens",
                    )
                answer_parts.append(delta)
        answer = "".join(answer_parts).strip()
        assert_runtime_snapshots_unchanged(self.retrieval, self.facts)
        if not answer:
            raise ProviderError(
                "EMPTY_DEEPSEEK_ANSWER",
                "DeepSeek returned no answer text",
                retryable=True,
            )
        if answer == refusal:
            for delta in answer_parts:
                yield SseAnswerDeltaEvent(delta=delta)
            yield SseDoneEvent(
                response=AnswerResponse(
                    request_id=str(uuid.uuid4()),
                    mode="direct",
                    answer_kind="insufficient_evidence",
                    answer=refusal,
                )
            )
            return
        cited_evidence, claims = bind_cited_evidence(
            answer, evidence_items, error_prefix="DIRECT"
        )
        for delta in answer_parts:
            yield SseAnswerDeltaEvent(delta=delta)
        yield SseDoneEvent(
            response=AnswerResponse(
                request_id=str(uuid.uuid4()),
                mode="direct",
                answer=answer,
                evidence=cited_evidence,
                claims=claims,
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
        raise ProviderError(
            "INVALID_SEARCH_LIMIT", "search limit must be between 1 and 50"
        )


def _diversify_hits(
    hits: Sequence[RetrievalHit], *, limit: int, per_document: int
) -> tuple[RetrievalHit, ...]:
    """Keep broad searches from being monopolized by one source document."""

    if limit <= 0 or per_document <= 0:
        raise ValueError("diversity limits must be positive")
    selected: list[RetrievalHit] = []
    document_counts: dict[str, int] = {}
    for hit in hits:
        locator = hit.locator
        if isinstance(locator, DocumentChunkLocator):
            count = document_counts.get(locator.document_id, 0)
            if count >= per_document:
                continue
            document_counts[locator.document_id] = count + 1
        selected.append(hit)
        if len(selected) == limit:
            break
    return tuple(selected)
