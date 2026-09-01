from __future__ import annotations

import asyncio
import json
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
from .direct import (
    LocalRetrieval,
    assert_runtime_snapshots_unchanged,
    build_recent_context,
    conservative_token_estimate,
)
from .grounding import bind_cited_evidence, insufficient_evidence_answer
from .providers import (
    AssistantToolTurn,
    ImageObserver,
    ProviderError,
    mark_images_uninspected,
)
from .stage2 import (
    coaching_state,
    finalize_stage2_answer,
    make_clarification_response,
    supports_stage2,
)

AGENTIC_INITIAL_HITS_PER_PROVIDER = 6
AGENTIC_MAX_SEARCH_RESULTS = 50
AGENTIC_MAX_READS = 24
AGENTIC_MAX_QUERY_LENGTH = 2_000
AGENTIC_MAX_INSPECT_IDS = 64
_FACT_RESOLUTION_ERROR_CODES = {
    "FACT_READ_ERROR",
    "INVALID_FACT_ARTIFACT",
    "LOCATOR_NOT_FOUND",
    "RETRIEVAL_SNAPSHOT_CHANGED",
    "STALE_LOCATOR",
}
_SEARCH_ERROR_CODES = {
    "INVALID_INDEX",
    "LEXICAL_INDEX_NOT_READY",
    "RETRIEVAL_SNAPSHOT_CHANGED",
    "SEMANTIC_INDEX_NOT_PRODUCTION_READY",
    "SEMANTIC_INDEX_NOT_READY",
    "SEMANTIC_PROVIDER_UNAVAILABLE",
    "ZHIPU_EMBEDDING_ERROR",
}

AGENTIC_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "search_keyword",
            "description": "Search local FTS5. Returns ranks and locators only, never facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": AGENTIC_MAX_QUERY_LENGTH,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": AGENTIC_MAX_SEARCH_RESULTS,
                    },
                },
                "required": ["query", "limit"],
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
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": AGENTIC_MAX_QUERY_LENGTH,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": AGENTIC_MAX_SEARCH_RESULTS,
                    },
                },
                "required": ["query", "limit"],
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
                        "items": {"type": "string", "maxLength": 1024},
                        "minItems": 1,
                        "maxItems": AGENTIC_MAX_INSPECT_IDS,
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
        max_reads: int = AGENTIC_MAX_READS,
        max_evidence_tokens: int = 200_000,
        max_search_calls: int = 12,
        max_tool_calls: int = 48,
        max_answer_tokens: int = 64_000,
        max_tool_output_tokens: int = 300_000,
        max_provider_output_tokens: int = 96_000,
        max_tool_argument_tokens: int = 32_000,
        timeout_seconds: float = 120.0,
    ):
        if (
            min(
                max_rounds,
                max_reads,
                max_evidence_tokens,
                max_search_calls,
                max_tool_calls,
                max_answer_tokens,
                max_tool_output_tokens,
                max_provider_output_tokens,
                max_tool_argument_tokens,
            )
            <= 0
            or timeout_seconds <= 0
        ):
            raise ValueError("agentic budgets must be positive")
        self.deepseek = deepseek
        self.retrieval = retrieval
        self.facts = facts
        self.system_prompt = system_prompt.strip()
        self.image_observer = image_observer
        self.max_rounds = max_rounds
        self.max_reads = max_reads
        self.max_evidence_tokens = max_evidence_tokens
        self.max_search_calls = max_search_calls
        self.max_tool_calls = max_tool_calls
        self.max_answer_tokens = max_answer_tokens
        self.max_tool_output_tokens = max_tool_output_tokens
        self.max_provider_output_tokens = max_provider_output_tokens
        self.max_tool_argument_tokens = max_tool_argument_tokens
        self.timeout_seconds = timeout_seconds

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
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        context = build_recent_context(history, current_question=request.question)
        stage2_enabled = supports_stage2(self.deepseek)
        research_question = request.question
        initial_keyword_query = request.question
        initial_semantic_query = request.question
        if stage2_enabled:
            completed_rounds, coaching_id = coaching_state(
                history,
                coaching_id=request.coaching_id,
                reply_to_message_id=request.reply_to_message_id,
            )
            yield SsePhaseEvent(phase="plan", message="正在判断问题和所需信息")
            plan = await _before_deadline(
                self.deepseek.plan_question(
                    question=request.question,
                    conversation_context=context,
                    completed_clarification_rounds=completed_rounds,
                    skip_clarification=request.skip_clarification,
                ),
                deadline,
            )
            if (
                plan.needs_clarification
                and not request.skip_clarification
                and completed_rounds < 3
            ):
                response = make_clarification_response(
                    request_id=str(uuid.uuid4()),
                    mode="agentic",
                    plan=plan,
                    completed_rounds=completed_rounds,
                    coaching_id=coaching_id,
                )
                yield SseAnswerDeltaEvent(delta=response.answer)
                yield SseDoneEvent(response=response)
                return
            research_question = plan.standalone_question
            initial_keyword_query = " ".join(
                (*plan.lexical_phrases_en, *plan.lexical_terms_en)
            ) or research_question
            initial_semantic_query = plan.semantic_query_en
        yield SsePhaseEvent(
            phase="retrieve",
            message="正在为智能检索准备初始候选结果",
        )
        initial_results, provider_counts = await _before_deadline(
            self._initial_search(
                initial_keyword_query, semantic_query=initial_semantic_query
            ),
            deadline,
        )
        assert_runtime_snapshots_unchanged(self.retrieval, self.facts)
        if all(
            "error" in initial_results[provider] for provider in ("keyword", "vector")
        ):
            raise ProviderError(
                "RETRIEVAL_UNAVAILABLE",
                "All configured retrieval providers failed",
                retryable=True,
            )
        refusal = insufficient_evidence_answer(request.question)
        system_prompt = self.system_prompt
        if stage2_enabled:
            system_prompt += (
                "\n\nSTAGE 2 RESEARCH HANDOFF: Use tools to gather a focused, diverse "
                "set of Evidence for the question. Do not write the user-facing answer. "
                "When research is complete, return a short JSON object such as "
                '{"research_complete":true}. A separate writer and verifier will compose '
                "and check the answer."
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "CONVERSATION CONTEXT (interpretation only; not evidence):\n"
                    f"{context or '[none]'}\n\nCURRENT QUESTION:\n{research_question}\n\n"
                    "INITIAL RETRIEVAL HITS (unfused locators/ranks only; not evidence):\n"
                    + json.dumps(
                        initial_results,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + (
                        "\n\nReturn a research-complete JSON handoff after tool use."
                        if stage2_enabled
                        else "\n\nINSUFFICIENT EVIDENCE CONTRACT:\n"
                        "If the read Evidence does not support an answer, return exactly "
                        "this text with no citation marker:\n" + refusal
                    )
                ),
            },
        ]
        evidence_by_id: OrderedDict[str, Evidence] = OrderedDict()
        evidence_tokens = 0
        read_count = 0
        search_count = 2
        tool_count = 0
        tool_output_tokens = 0
        provider_output_tokens = 0
        tool_output_exhausted = False
        fact_resolution_failures: list[str] = []
        yield SsePhaseEvent(phase="agentic", message="智能体正在选择并读取原文证据")
        final_answer: str | None = None
        for _round in range(self.max_rounds):
            turn = await _before_deadline(
                self.deepseek.tool_turn(messages=messages, tools=AGENTIC_TOOLS),
                deadline,
            )
            try:
                provider_message_json = json.dumps(
                    turn.provider_message,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise ProviderError(
                    "INVALID_DEEPSEEK_TOOL_RESPONSE",
                    "DeepSeek returned an invalid tool response",
                ) from exc
            provider_output_tokens += conservative_token_estimate(provider_message_json)
            if provider_output_tokens > self.max_provider_output_tokens:
                raise ProviderError(
                    "AGENTIC_PROVIDER_MESSAGE_LIMIT",
                    "Agentic provider output exceeded its cumulative budget",
                )
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
                tool_count += 1
                if tool_count > self.max_tool_calls:
                    raise ProviderError(
                        "AGENTIC_TOOL_LIMIT",
                        f"Agentic tool budget exceeds {self.max_tool_calls} calls",
                    )
                if tool_output_exhausted:
                    raise ProviderError(
                        "AGENTIC_TOOL_OUTPUT_LIMIT",
                        "Agentic tool output budget was already exhausted",
                    )
                evidence_before = evidence_by_id.copy()
                evidence_tokens_before = evidence_tokens
                if (
                    conservative_token_estimate(call.arguments)
                    > self.max_tool_argument_tokens
                ):
                    result: dict[str, Any] = _tool_error(
                        "AGENTIC_TOOL_ARGUMENT_LIMIT",
                        "Tool arguments exceed the allowed size",
                    )
                else:
                    try:
                        arguments = json.loads(call.arguments)
                        if not isinstance(arguments, dict):
                            raise TypeError("tool arguments must be an object")
                    except (json.JSONDecodeError, TypeError, ValueError):
                        result = _tool_error(
                            "INVALID_TOOL_ARGUMENTS",
                            "tool arguments must be a valid JSON object",
                        )
                    else:
                        if call.name == "search_keyword":
                            search_count += 1
                            if search_count > self.max_search_calls:
                                raise ProviderError(
                                    "AGENTIC_SEARCH_LIMIT",
                                    f"Agentic search budget exceeds {self.max_search_calls} calls",
                                )
                            result = await _before_deadline(
                                self._search("keyword", arguments, provider_counts),
                                deadline,
                            )
                        elif call.name == "search_semantic":
                            search_count += 1
                            if search_count > self.max_search_calls:
                                raise ProviderError(
                                    "AGENTIC_SEARCH_LIMIT",
                                    f"Agentic search budget exceeds {self.max_search_calls} calls",
                                )
                            result = await _before_deadline(
                                self._search("vector", arguments, provider_counts),
                                deadline,
                            )
                        elif call.name == "read_source_window":
                            read_count += 1
                            if read_count > self.max_reads:
                                result = _tool_error(
                                    "AGENTIC_READ_LIMIT",
                                    f"Read budget reached ({self.max_reads}). Stop calling tools "
                                    "and answer from evidence already read; if it is insufficient, say so.",
                                )
                            else:
                                result, added_tokens = await _before_deadline(
                                    self._read(
                                        arguments,
                                        evidence_by_id,
                                        evidence_tokens=evidence_tokens,
                                    ),
                                    deadline,
                                )
                                evidence_tokens += added_tokens
                                raw_error = result.get("error")
                                if isinstance(raw_error, dict):
                                    error_code = raw_error.get("code")
                                    if error_code in _FACT_RESOLUTION_ERROR_CODES:
                                        fact_resolution_failures.append(str(error_code))
                        elif call.name == "inspect_images":
                            result, added_tokens = await _before_deadline(
                                self._inspect(
                                    arguments,
                                    evidence_by_id,
                                    evidence_tokens=evidence_tokens,
                                ),
                                deadline,
                            )
                            evidence_tokens += added_tokens
                        else:
                            result = _tool_error(
                                "UNKNOWN_TOOL", f"Unknown tool: {call.name}"
                            )
                serialized_result = json.dumps(
                    result, ensure_ascii=False, separators=(",", ":")
                )
                result_tokens = conservative_token_estimate(serialized_result)
                if tool_output_tokens + result_tokens > self.max_tool_output_tokens:
                    evidence_by_id.clear()
                    evidence_by_id.update(evidence_before)
                    evidence_tokens = evidence_tokens_before
                    tool_output_exhausted = True
                    result = _tool_error(
                        "AGENTIC_TOOL_OUTPUT_LIMIT",
                        "Tool output budget reached. Stop calling tools and answer "
                        "from Evidence already read.",
                    )
                    serialized_result = json.dumps(
                        result, ensure_ascii=False, separators=(",", ":")
                    )
                else:
                    tool_output_tokens += result_tokens
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": serialized_result,
                    }
                )
        else:
            raise ProviderError(
                "AGENTIC_ROUND_LIMIT",
                f"Agentic tool loop exceeds {self.max_rounds} rounds",
            )

        assert final_answer is not None
        assert_runtime_snapshots_unchanged(self.retrieval, self.facts)
        if not evidence_by_id and tool_output_exhausted:
            raise ProviderError(
                "AGENTIC_TOOL_OUTPUT_LIMIT",
                "Agentic tool output budget was exhausted before evidence was read",
            )
        if not evidence_by_id and fact_resolution_failures:
            raise ProviderError(
                "FACT_RESOLUTION_FAILED",
                "Requested source candidates could not be resolved against the facts snapshot",
                retryable=True,
            )
        if conservative_token_estimate(final_answer) > self.max_answer_tokens:
            raise ProviderError(
                "AGENTIC_ANSWER_LIMIT",
                f"Answer exceeds {self.max_answer_tokens} conservative tokens",
            )
        if stage2_enabled:
            yield SsePhaseEvent(phase="answer", message="正在组织结论和建议")
            evidence_items = tuple(evidence_by_id.values())
            draft = await _before_deadline(
                self.deepseek.draft_stage2_answer(
                    question=request.question,
                    conversation_context=context,
                    evidence=evidence_items,
                    message_ids=tuple(message.id for message in history),
                ),
                deadline,
            )
            yield SsePhaseEvent(
                phase="verify",
                message="正在独立核验每条结论，必要时自动修复一次",
            )
            response = await _before_deadline(
                finalize_stage2_answer(
                    provider=self.deepseek,
                    mode="agentic",
                    request_id=str(uuid.uuid4()),
                    question=request.question,
                    conversation_context=context,
                    evidence=evidence_items,
                    history=history,
                    initial_draft=draft,
                ),
                deadline,
            )
            assert_runtime_snapshots_unchanged(self.retrieval, self.facts)
            if conservative_token_estimate(response.answer) > self.max_answer_tokens:
                raise ProviderError(
                    "AGENTIC_ANSWER_LIMIT",
                    f"Answer exceeds {self.max_answer_tokens} conservative tokens",
                )
            yield SseRetrievalSummaryEvent(
                provider_counts=provider_counts,
                fused_count=0,
            )
            for item in response.evidence:
                yield SseEvidenceEvent(evidence=item)
            yield SseAnswerDeltaEvent(delta=response.answer)
            yield SseDoneEvent(response=response)
            return
        if final_answer == refusal:
            evidence = ()
            claims = ()
            answer_kind = "insufficient_evidence"
        elif evidence_by_id or "[evidence:" in final_answer:
            evidence, claims = bind_cited_evidence(
                final_answer,
                tuple(evidence_by_id.values()),
                error_prefix="AGENTIC",
            )
            answer_kind = "grounded"
        else:
            final_answer = refusal
            evidence = ()
            claims = ()
            answer_kind = "insufficient_evidence"
        yield SseRetrievalSummaryEvent(
            provider_counts=provider_counts,
            fused_count=0,
        )
        for item in evidence:
            yield SseEvidenceEvent(evidence=item)
        yield SsePhaseEvent(phase="answer", message="正在根据已读取的证据生成回答")
        yield SseAnswerDeltaEvent(delta=final_answer)
        yield SseDoneEvent(
            response=AnswerResponse(
                request_id=str(uuid.uuid4()),
                mode="agentic",
                answer_kind=answer_kind,
                answer=final_answer,
                evidence=evidence,
                claims=claims,
            )
        )

    async def _search(
        self,
        provider: str,
        arguments: dict[str, Any],
        provider_counts: dict[str, int],
    ) -> dict[str, Any]:
        query = arguments.get("query")
        limit = arguments.get("limit")
        if not isinstance(query, str) or not query.strip():
            return _tool_error("INVALID_SEARCH_QUERY", "query must be non-empty")
        if len(query) > AGENTIC_MAX_QUERY_LENGTH:
            return _tool_error(
                "INVALID_SEARCH_QUERY",
                f"query must contain at most {AGENTIC_MAX_QUERY_LENGTH} characters",
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= AGENTIC_MAX_SEARCH_RESULTS
        ):
            return _tool_error(
                "INVALID_SEARCH_LIMIT",
                f"limit must be 1..{AGENTIC_MAX_SEARCH_RESULTS}",
            )
        try:
            if provider == "keyword":
                hits = await self.retrieval.keyword(query, limit=limit)
            else:
                hits = await self.retrieval.semantic(query, limit=limit)
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            if getattr(exc, "code", None) == "RETRIEVAL_SNAPSHOT_CHANGED":
                raise ProviderError(
                    "RETRIEVAL_SNAPSHOT_CHANGED",
                    "The validated retrieval snapshot changed during the answer run",
                    retryable=True,
                ) from exc
            return _safe_exception_tool_error(
                exc,
                allowed_codes=_SEARCH_ERROR_CODES,
                fallback_code="SEARCH_ERROR",
                public_message="Local search could not complete the request",
            )
        provider_counts[provider] += len(hits)
        return {"hits": _serialize_hits(hits)}

    async def _initial_search(
        self, query: str, *, semantic_query: str | None = None
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Seed the first model turn without hidden rewriting or fusion."""

        keyword_result, vector_result = await asyncio.gather(
            self.retrieval.keyword(query, limit=AGENTIC_INITIAL_HITS_PER_PROVIDER),
            self.retrieval.semantic(
                semantic_query or query, limit=AGENTIC_INITIAL_HITS_PER_PROVIDER
            ),
            return_exceptions=True,
        )
        for result in (keyword_result, vector_result):
            if (
                isinstance(result, BaseException)
                and getattr(result, "code", None) == "RETRIEVAL_SNAPSHOT_CHANGED"
            ):
                raise ProviderError(
                    "RETRIEVAL_SNAPSHOT_CHANGED",
                    "The validated retrieval snapshot changed during the answer run",
                    retryable=True,
                ) from result
        results: dict[str, Any] = {}
        counts = {"keyword": 0, "vector": 0}
        for provider, result in (
            ("keyword", keyword_result),
            ("vector", vector_result),
        ):
            if isinstance(result, BaseException):
                results[provider] = {
                    "hits": [],
                    **_safe_exception_tool_error(
                        result,
                        allowed_codes=_SEARCH_ERROR_CODES,
                        fallback_code="SEARCH_ERROR",
                        public_message="Local search could not complete the request",
                    ),
                }
                continue
            counts[provider] = len(result)
            results[provider] = {"hits": _serialize_hits(result)}
        return results, counts

    async def _read(
        self,
        arguments: dict[str, Any],
        evidence_by_id: OrderedDict[str, Evidence],
        *,
        evidence_tokens: int,
    ) -> tuple[dict[str, Any], int]:
        window = arguments.get("window", 0)
        if (
            isinstance(window, bool)
            or not isinstance(window, int)
            or not 0 <= window <= 8
        ):
            raise ProviderError(
                "AGENTIC_INVALID_WINDOW", "source window must be between 0 and 8"
            )
        try:
            locator = DocumentChunkLocator.model_validate(arguments.get("locator"))
        except ValueError:
            return (
                _tool_error(
                    "INVALID_LOCATOR",
                    "locator does not match the required source schema",
                ),
                0,
            )
        try:
            result = await asyncio.to_thread(
                self.facts.read_source_window, locator, window
            )
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            if getattr(exc, "code", None) == "RETRIEVAL_SNAPSHOT_CHANGED":
                raise ProviderError(
                    "RETRIEVAL_SNAPSHOT_CHANGED",
                    "The validated retrieval snapshot changed during the answer run",
                    retryable=True,
                ) from exc
            return (
                _safe_exception_tool_error(
                    exc,
                    allowed_codes=_FACT_RESOLUTION_ERROR_CODES,
                    fallback_code="FACT_READ_ERROR",
                    public_message=(
                        "Source evidence could not be read from the validated snapshot"
                    ),
                ),
                0,
            )
        additions = [item for item in result if item.evidence_id not in evidence_by_id]
        already_read = [
            item.evidence_id for item in result if item.evidence_id in evidence_by_id
        ]
        added_tokens = sum(
            conservative_token_estimate(item.model_dump_json()) for item in additions
        )
        if evidence_tokens + added_tokens > self.max_evidence_tokens:
            return (
                _tool_error(
                    "AGENTIC_EVIDENCE_LIMIT",
                    f"Evidence budget reached ({self.max_evidence_tokens} conservative "
                    "tokens). Stop calling tools and answer from evidence already read; "
                    "if it is insufficient, say so.",
                ),
                0,
            )
        for item in additions:
            evidence_by_id[item.evidence_id] = item
        response: dict[str, Any] = {
            "evidence": [item.model_dump(mode="json") for item in additions]
        }
        if already_read:
            response["already_read"] = already_read
        return response, added_tokens

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
            or len(raw_ids) > AGENTIC_MAX_INSPECT_IDS
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
                warning = _tool_error("VISION_UNAVAILABLE", "vision inspection failed")
        old_tokens = sum(
            conservative_token_estimate(item.model_dump_json()) for item in selected
        )
        new_tokens = sum(
            conservative_token_estimate(item.model_dump_json()) for item in enriched
        )
        added_tokens = max(0, new_tokens - old_tokens)
        if evidence_tokens + added_tokens > self.max_evidence_tokens:
            return (
                _tool_error(
                    "AGENTIC_EVIDENCE_LIMIT",
                    f"Evidence budget reached ({self.max_evidence_tokens} conservative "
                    "tokens). Stop calling tools and answer from evidence already read; "
                    "if it is insufficient, say so.",
                ),
                0,
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
    agentic_available = True

    def __init__(
        self,
        *,
        direct: Any,
        agentic: AgenticAnswerService,
    ):
        self.direct = direct
        self.agentic = agentic

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
        return self.agentic.stream_answer(
            conversation_id=conversation_id, request=request, history=history
        )


def _tool_error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _safe_exception_tool_error(
    exc: BaseException,
    *,
    allowed_codes: set[str],
    fallback_code: str,
    public_message: str,
) -> dict[str, Any]:
    raw_code = getattr(exc, "code", None)
    code = raw_code if isinstance(raw_code, str) and raw_code in allowed_codes else None
    return _tool_error(code or fallback_code, public_message)


async def _before_deadline(awaitable: Any, deadline: float) -> Any:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise ProviderError(
            "AGENTIC_DEADLINE",
            "Agentic answer exceeded its total deadline",
            retryable=True,
        )
    try:
        return await asyncio.wait_for(awaitable, timeout=remaining)
    except TimeoutError as exc:
        raise ProviderError(
            "AGENTIC_DEADLINE",
            "Agentic answer exceeded its total deadline",
            retryable=True,
        ) from exc


def _serialize_hits(hits: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "provider": str(hit.provider),
            "rank": hit.rank,
            "locator": hit.locator.model_dump(mode="json"),
        }
        for hit in hits
    ]
