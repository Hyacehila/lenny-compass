from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

from ai_pm_contracts import (
    AnswerRequest,
    AnswerResponse,
    DocumentChunkLocator,
    Evidence,
    SseAnswerDeltaEvent,
    SseDoneEvent,
    SseEvent,
    SseEvidenceEvent,
    SsePhaseEvent,
    SseRetrievalSummaryEvent,
)

from .conversations import Message
from .direct import LocalRetrieval, build_recent_context, conservative_token_estimate
from .providers import (
    AssistantToolTurn,
    ImageObserver,
    ProviderError,
    mark_images_uninspected,
)

AGENTIC_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "search_keyword",
            "description": "Search local FTS5. Returns ranks and locators only, never facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic",
            "description": "Search local 2048d vectors. Returns ranks and locators only, never facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source_window",
            "description": "Resolve one returned document locator into typed source Evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "locator": {"type": "object"},
                    "window": {"type": "integer", "minimum": 0, "maximum": 8},
                },
                "required": ["locator"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_images",
            "description": "Inspect remote images belonging to Evidence already read this turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    }
                },
                "required": ["evidence_ids"],
                "additionalProperties": False,
            },
        },
    },
)


class ToolModel(Protocol):
    async def tool_turn(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantToolTurn: ...


class FactReader(Protocol):
    def read_source_window(
        self, locator: DocumentChunkLocator, window: int = 0
    ) -> tuple[Evidence, ...]: ...


class AgenticAnswerService:
    def __init__(
        self,
        *,
        deepseek: ToolModel,
        retrieval: LocalRetrieval,
        facts: FactReader,
        system_prompt: str,
        image_observer: ImageObserver | None = None,
        max_rounds: int = 16,
        max_reads: int = 12,
        max_evidence_tokens: int = 200_000,
    ):
        self.deepseek = deepseek
        self.retrieval = retrieval
        self.facts = facts
        self.system_prompt = system_prompt.strip()
        self.image_observer = image_observer
        self.max_rounds = max_rounds
        self.max_reads = max_reads
        self.max_evidence_tokens = max_evidence_tokens

    async def stream_answer(
        self,
        *,
        conversation_id: str,
        request: AnswerRequest,
        history: Sequence[Message],
    ) -> AsyncIterator[SseEvent]:
        if request.mode != "agentic":
            raise ProviderError(
                "INVALID_ANSWER_MODE", "AgenticAnswerService requires agentic mode"
            )
        context = build_recent_context(history, current_question=request.question)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "CONVERSATION CONTEXT (interpretation only; not evidence):\n"
                    f"{context or '[none]'}\n\nCURRENT QUESTION:\n{request.question}"
                ),
            },
        ]
        evidence_by_id: OrderedDict[str, Evidence] = OrderedDict()
        evidence_tokens = 0
        read_count = 0
        provider_counts = {"keyword": 0, "vector": 0}

        yield SsePhaseEvent(
            phase="agentic", message="Selecting and reading source evidence"
        )
        final_answer: str | None = None
        for _round in range(self.max_rounds):
            turn = await self.deepseek.tool_turn(messages=messages, tools=AGENTIC_TOOLS)
            if not turn.tool_calls:
                final_answer = (turn.content or "").strip()
                if not final_answer:
                    raise ProviderError(
                        "EMPTY_DEEPSEEK_ANSWER", "DeepSeek returned no answer text"
                    )
                break

            # This carries reasoning_content back byte-for-byte/value-for-value,
            # but the in-memory messages are never persisted or sent over SSE.
            messages.append(turn.provider_message)
            for call in turn.tool_calls:
                try:
                    arguments = json.loads(call.arguments)
                    if not isinstance(arguments, dict):
                        raise TypeError("tool arguments must be an object")
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    result: dict[str, Any] = _tool_error(
                        "INVALID_TOOL_ARGUMENTS", str(exc)
                    )
                else:
                    if call.name == "search_keyword":
                        result = await self._search(
                            "keyword", arguments, provider_counts
                        )
                    elif call.name == "search_semantic":
                        result = await self._search(
                            "vector", arguments, provider_counts
                        )
                    elif call.name == "read_source_window":
                        read_count += 1
                        if read_count > self.max_reads:
                            raise ProviderError(
                                "AGENTIC_READ_LIMIT",
                                f"Agentic fact reads exceed {self.max_reads}",
                            )
                        result, added_tokens = self._read(
                            arguments,
                            evidence_by_id,
                            evidence_tokens=evidence_tokens,
                        )
                        evidence_tokens += added_tokens
                    elif call.name == "inspect_images":
                        result, added_tokens = await self._inspect(
                            arguments,
                            evidence_by_id,
                            evidence_tokens=evidence_tokens,
                        )
                        evidence_tokens += added_tokens
                    else:
                        result = _tool_error(
                            "UNKNOWN_TOOL", f"Unknown tool: {call.name}"
                        )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            result, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
        else:
            raise ProviderError(
                "AGENTIC_ROUND_LIMIT",
                f"Agentic tool loop exceeds {self.max_rounds} rounds",
            )

        assert final_answer is not None
        _validate_citations(final_answer, set(evidence_by_id))
        evidence = tuple(evidence_by_id.values())
        yield SseRetrievalSummaryEvent(
            provider_counts=provider_counts,
            fused_count=0,
        )
        for item in evidence:
            yield SseEvidenceEvent(evidence=item)
        yield SsePhaseEvent(phase="answer", message="Answering from read evidence")
        yield SseAnswerDeltaEvent(delta=final_answer)
        yield SseDoneEvent(
            response=AnswerResponse(
                request_id=str(uuid.uuid4()),
                mode="agentic",
                answer=final_answer,
                evidence=evidence,
            )
        )

    async def _search(
        self,
        provider: str,
        arguments: dict[str, Any],
        provider_counts: dict[str, int],
    ) -> dict[str, Any]:
        query = arguments.get("query")
        limit = arguments.get("limit", 50)
        if not isinstance(query, str) or not query.strip():
            return _tool_error("INVALID_SEARCH_QUERY", "query must be non-empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            return _tool_error("INVALID_SEARCH_LIMIT", "limit must be 1..50")
        try:
            if provider == "keyword":
                hits = await self.retrieval.keyword(query, limit=limit)
            else:
                hits = await self.retrieval.semantic(query, limit=limit)
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            return _tool_error(
                str(getattr(exc, "code", "SEARCH_ERROR")), str(exc)
            )
        provider_counts[provider] += len(hits)
        return {
            "hits": [
                {
                    "provider": str(hit.provider),
                    "rank": hit.rank,
                    "locator": hit.locator.model_dump(mode="json"),
                }
                for hit in hits
            ]
        }

    def _read(
        self,
        arguments: dict[str, Any],
        evidence_by_id: OrderedDict[str, Evidence],
        *,
        evidence_tokens: int,
    ) -> tuple[dict[str, Any], int]:
        window = arguments.get("window", 0)
        if isinstance(window, bool) or not isinstance(window, int) or not 0 <= window <= 8:
            raise ProviderError(
                "AGENTIC_INVALID_WINDOW", "source window must be between 0 and 8"
            )
        try:
            locator = DocumentChunkLocator.model_validate(arguments.get("locator"))
        except ValueError as exc:
            return _tool_error("INVALID_LOCATOR", str(exc)), 0
        try:
            result = self.facts.read_source_window(locator, window)
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            return (
                _tool_error(str(getattr(exc, "code", "FACT_READ_ERROR")), str(exc)),
                0,
            )
        additions = [item for item in result if item.evidence_id not in evidence_by_id]
        added_tokens = sum(
            conservative_token_estimate(item.model_dump_json()) for item in additions
        )
        if evidence_tokens + added_tokens > self.max_evidence_tokens:
            raise ProviderError(
                "AGENTIC_EVIDENCE_LIMIT",
                f"Agentic evidence exceeds {self.max_evidence_tokens} conservative tokens",
            )
        for item in additions:
            evidence_by_id[item.evidence_id] = item
        return (
            {"evidence": [item.model_dump(mode="json") for item in result]},
            added_tokens,
        )

    async def _inspect(
        self,
        arguments: dict[str, Any],
        evidence_by_id: OrderedDict[str, Evidence],
        *,
        evidence_tokens: int,
    ) -> tuple[dict[str, Any], int]:
        raw_ids = arguments.get("evidence_ids")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(item, str) for item in raw_ids)
        ):
            return (
                _tool_error(
                    "INVALID_EVIDENCE_IDS",
                    "evidence_ids must be a non-empty string array",
                ),
                0,
            )
        unknown = [item for item in raw_ids if item not in evidence_by_id]
        if unknown:
            return (
                _tool_error(
                    "EVIDENCE_NOT_READ",
                    "inspect_images accepts only Evidence read in this turn",
                ),
                0,
            )
        selected = tuple(evidence_by_id[item] for item in raw_ids)
        warning: dict[str, Any] | None = None
        if self.image_observer is None:
            enriched = mark_images_uninspected(
                selected, error="ZHIPU_VISION_NOT_CONFIGURED"
            )
            warning = _tool_error(
                "VISION_UNAVAILABLE", "vision provider is not configured"
            )
        else:
            try:
                enriched = await self.image_observer.enrich(selected)
            except (OSError, RuntimeError, ValueError, sqlite3.Error):
                enriched = mark_images_uninspected(
                    selected, error="ZHIPU_VISION_UNAVAILABLE"
                )
                warning = _tool_error(
                    "VISION_UNAVAILABLE", "vision inspection failed"
                )
        old_tokens = sum(
            conservative_token_estimate(item.model_dump_json()) for item in selected
        )
        new_tokens = sum(
            conservative_token_estimate(item.model_dump_json()) for item in enriched
        )
        added_tokens = max(0, new_tokens - old_tokens)
        if evidence_tokens + added_tokens > self.max_evidence_tokens:
            raise ProviderError(
                "AGENTIC_EVIDENCE_LIMIT",
                f"Agentic evidence exceeds {self.max_evidence_tokens} conservative tokens",
            )
        for item in enriched:
            evidence_by_id[item.evidence_id] = item
        result: dict[str, Any] = {
            "image_observations": [
                observation.model_dump(mode="json")
                for item in enriched
                for observation in item.image_observations
            ]
        }
        if warning is not None:
            result["warning"] = warning
        return result, added_tokens


class AnswerRouter:
    def __init__(
        self,
        *,
        direct: Any,
        agentic: AgenticAnswerService | None,
        agentic_enabled: bool,
    ):
        self.direct = direct
        self.agentic = agentic
        self.agentic_enabled = agentic_enabled

    def stream_answer(
        self,
        *,
        conversation_id: str,
        request: AnswerRequest,
        history: Sequence[Message],
    ) -> AsyncIterator[SseEvent]:
        if request.mode == "direct":
            return self.direct.stream_answer(
                conversation_id=conversation_id, request=request, history=history
            )
        if not self.agentic_enabled or self.agentic is None:
            return _raise_agentic_disabled()
        return self.agentic.stream_answer(
            conversation_id=conversation_id, request=request, history=history
        )


async def _raise_agentic_disabled() -> AsyncIterator[SseEvent]:
    raise ProviderError(
        "AGENTIC_RAG_DISABLED", "Agentic RAG is disabled by configuration"
    )
    if False:  # pragma: no cover - makes this an async iterator for the protocol
        yield SsePhaseEvent(phase="unreachable")


def _tool_error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


_CITATION = re.compile(r"\[evidence:([^\]]+)\]")


def _validate_citations(answer: str, allowed_ids: set[str]) -> None:
    unread = sorted(set(_CITATION.findall(answer)) - allowed_ids)
    if unread:
        raise ProviderError(
            "AGENTIC_UNREAD_CITATION",
            "Answer cited Evidence that was not read in this turn",
        )
