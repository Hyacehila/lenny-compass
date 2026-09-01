from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from ai_pm_contracts import (
    DocumentChunkLocator,
    Evidence,
    ImageObservation,
    QueryRewrite,
)
from pydantic import BaseModel, ConfigDict, Field


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _require_finish_reason(
    choice: Any,
    *,
    allowed: set[str],
    code_prefix: str,
) -> str:
    if not isinstance(choice, dict):
        raise ProviderError(
            f"{code_prefix}_INVALID_FINISH",
            "Provider response choice is invalid",
        )
    reason = choice.get("finish_reason")
    if reason in allowed:
        return str(reason)
    if reason == "length":
        raise ProviderError(
            f"{code_prefix}_TRUNCATED",
            "Provider output was truncated by its generation limit",
        )
    if reason == "insufficient_system_resource":
        raise ProviderError(
            f"{code_prefix}_RESOURCE_EXHAUSTED",
            "Provider could not complete the response due to resource limits",
            retryable=True,
        )
    raise ProviderError(
        f"{code_prefix}_INVALID_FINISH",
        "Provider response ended without an allowed finish reason",
    )


class _RewritePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    semantic_query_en: str = Field(min_length=1)
    lexical_terms_en: tuple[str, ...] = ()
    lexical_phrases_en: tuple[str, ...] = ()


_FALLBACK_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*|\d+(?:\.\d+)?")


def fallback_query_rewrite(question: str) -> QueryRewrite:
    """Lossless fallback: retain the original rather than invent a translation."""

    terms: list[str] = []
    seen: set[str] = set()
    for value in _FALLBACK_TOKEN.findall(question):
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            terms.append(value)
        if len(terms) == 8:
            break
    if not terms:
        # unicode61 can still search this verbatim; calling it English would be unsafe.
        terms.append(question)
    return QueryRewrite(
        original_query=question,
        semantic_query_en=question,
        lexical_terms_en=tuple(terms),
        lexical_phrases_en=(),
        used_fallback=True,
    )


class DeepSeekClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        rewrite_model: str,
        answer_model: str,
        rewrite_prompt: str,
        answer_prompt: str,
        plan_prompt: str = "",
        stage2_answer_prompt: str = "",
        verify_prompt: str = "",
        repair_prompt: str = "",
        verify_model: str | None = None,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        tool_max_tokens: int = 32_000,
        answer_max_tokens: int = 64_000,
    ):
        if not api_key:
            raise ProviderError(
                "MISSING_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY is not set"
            )
        if tool_max_tokens <= 0 or answer_max_tokens <= 0:
            raise ValueError("DeepSeek output token limits must be positive")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.rewrite_model = rewrite_model
        self.answer_model = answer_model
        self.rewrite_prompt = rewrite_prompt.strip()
        self.answer_prompt = answer_prompt.strip()
        self.plan_prompt = plan_prompt.strip()
        self.stage2_answer_prompt = stage2_answer_prompt.strip()
        self.verify_prompt = verify_prompt.strip()
        self.repair_prompt = repair_prompt.strip()
        self.verify_model = verify_model or answer_model
        self.timeout = timeout
        self.transport = transport
        self.tool_max_tokens = tool_max_tokens
        self.answer_max_tokens = answer_max_tokens

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def rewrite(self, question: str, conversation_context: str) -> QueryRewrite:
        payload = {
            "model": self.rewrite_model,
            "messages": [
                {"role": "system", "content": self.rewrite_prompt},
                {
                    "role": "user",
                    "content": (
                        "CONVERSATION CONTEXT (reference resolution only):\n"
                        f"{conversation_context or '[none]'}\n\n"
                        f"CURRENT QUESTION:\n{question}"
                    ),
                },
            ],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
            "response_format": {"type": "json_object"},
            "max_tokens": 1_024,
            "stream": False,
        }
        try:
            body = await self._post_json(payload, code="DEEPSEEK_REWRITE_ERROR")
            choice = body["choices"][0]
            _require_finish_reason(
                choice,
                allowed={"stop"},
                code_prefix="DEEPSEEK_REWRITE",
            )
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("rewrite content is not a string")
            parsed = _RewritePayload.model_validate(json.loads(content))
            return QueryRewrite(
                original_query=question,
                semantic_query_en=parsed.semantic_query_en,
                lexical_terms_en=_unique_nonempty(parsed.lexical_terms_en, limit=8),
                lexical_phrases_en=_unique_nonempty(parsed.lexical_phrases_en, limit=8),
                used_fallback=False,
            )
        except (IndexError, KeyError, ProviderError, TypeError, ValueError):
            return fallback_query_rewrite(question)

    async def plan_question(
        self,
        *,
        question: str,
        conversation_context: str,
        completed_clarification_rounds: int,
        skip_clarification: bool,
    ) -> Any:
        from .stage2 import QuestionPlan

        payload = {
            "model": self.rewrite_model,
            "messages": [
                {"role": "system", "content": self.plan_prompt},
                {
                    "role": "user",
                    "content": (
                        f"CURRENT QUESTION:\n{question}\n\n"
                        "RECENT CONVERSATION (interpretation only):\n"
                        f"{conversation_context or '[none]'}\n\n"
                        "COACHING STATE:\n"
                        f"completed_rounds={completed_clarification_rounds}\n"
                        f"skip_clarification={str(skip_clarification).lower()}"
                    ),
                },
            ],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
            "response_format": {"type": "json_object"},
            "max_tokens": 2_048,
            "stream": False,
        }
        try:
            body = await self._post_json(payload, code="DEEPSEEK_PLAN_ERROR")
            content = _completion_content(body, code_prefix="DEEPSEEK_PLAN")
            return QuestionPlan.model_validate(json.loads(content))
        except (ProviderError, TypeError, ValueError, json.JSONDecodeError):
            rewrite = fallback_query_rewrite(question)
            return QuestionPlan(
                intent="follow_up" if conversation_context else "knowledge",
                standalone_question=question,
                semantic_query_en=rewrite.semantic_query_en,
                lexical_terms_en=rewrite.lexical_terms_en,
                lexical_phrases_en=rewrite.lexical_phrases_en,
            )

    async def draft_stage2_answer(
        self,
        *,
        question: str,
        conversation_context: str,
        evidence: Sequence[Evidence],
        message_ids: Sequence[str],
    ) -> Any:
        from .stage2 import AnswerDraft

        payload = {
            "model": self.answer_model,
            "messages": [
                {"role": "system", "content": self.stage2_answer_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "conversation_context": conversation_context,
                            "available_message_ids": list(message_ids),
                            "resolved_evidence": _stage2_evidence(evidence),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
            "response_format": {"type": "json_object"},
            "max_tokens": self.answer_max_tokens,
            "stream": False,
        }
        try:
            body = await self._post_json(payload, code="DEEPSEEK_ANSWER_ERROR")
            content = _completion_content(body, code_prefix="DEEPSEEK_STAGE2_DRAFT")
            return AnswerDraft.model_validate(json.loads(content))
        except ProviderError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "INVALID_DEEPSEEK_STAGE2_DRAFT",
                "DeepSeek returned an invalid structured answer",
            ) from exc

    async def verify_stage2_answer(
        self,
        *,
        draft: Any,
        evidence: Sequence[Evidence],
        conversation_context: str,
    ) -> Any:
        from .stage2 import VerificationBatch

        payload = {
            "model": self.verify_model,
            "messages": [
                {"role": "system", "content": self.verify_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "draft": draft.model_dump(mode="json"),
                            "resolved_evidence": _stage2_evidence(evidence),
                            "user_context": conversation_context,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
            "response_format": {"type": "json_object"},
            "max_tokens": 8_192,
            "stream": False,
        }
        try:
            body = await self._post_json(
                payload, code="SEMANTIC_VERIFICATION_UNAVAILABLE"
            )
            content = _completion_content(
                body, code_prefix="SEMANTIC_VERIFICATION"
            )
            return VerificationBatch.model_validate(json.loads(content))
        except ProviderError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "SEMANTIC_VERIFICATION_INVALID_RESPONSE",
                "Semantic verifier returned an invalid result",
                retryable=True,
            ) from exc

    async def repair_stage2_answer(
        self,
        *,
        question: str,
        draft: Any,
        verification: Any,
        evidence: Sequence[Evidence],
        conversation_context: str,
        message_ids: Sequence[str],
    ) -> Any:
        from .stage2 import AnswerDraft

        payload = {
            "model": self.answer_model,
            "messages": [
                {"role": "system", "content": self.repair_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "conversation_context": conversation_context,
                            "available_message_ids": list(message_ids),
                            "draft": draft.model_dump(mode="json"),
                            "verification": verification.model_dump(mode="json"),
                            "resolved_evidence": _stage2_evidence(evidence),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
            "response_format": {"type": "json_object"},
            "max_tokens": self.answer_max_tokens,
            "stream": False,
        }
        try:
            body = await self._post_json(payload, code="DEEPSEEK_REPAIR_ERROR")
            content = _completion_content(body, code_prefix="DEEPSEEK_REPAIR")
            return AnswerDraft.model_validate(json.loads(content))
        except ProviderError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "INVALID_DEEPSEEK_REPAIR",
                "DeepSeek returned an invalid repaired answer",
            ) from exc

    async def tool_turn(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantToolTurn:
        payload = {
            "model": self.answer_model,
            "messages": list(messages),
            "tools": list(tools),
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
            "max_tokens": self.tool_max_tokens,
            "stream": False,
        }
        body = await self._post_json(payload, code="DEEPSEEK_AGENTIC_ERROR")
        try:
            choice = body["choices"][0]
            raw_message = choice["message"]
            if not isinstance(raw_message, dict):
                raise TypeError("assistant message is not an object")
            content = raw_message.get("content")
            if content is not None and not isinstance(content, str):
                raise TypeError("assistant content is not a string")
            raw_calls = raw_message.get("tool_calls", [])
            if raw_calls is None:
                raw_calls = []
            if not isinstance(raw_calls, list):
                raise TypeError("tool_calls is not an array")
            calls: list[ToolCall] = []
            for raw in raw_calls:
                if not isinstance(raw, dict):
                    raise TypeError("tool call is not an object")
                function = raw["function"]
                if not isinstance(function, dict):
                    raise TypeError("tool function is not an object")
                calls.append(
                    ToolCall(
                        id=str(raw["id"]),
                        name=str(function["name"]),
                        arguments=str(function["arguments"]),
                    )
                )
            _require_finish_reason(
                choice,
                allowed={"tool_calls"} if calls else {"stop"},
                code_prefix="DEEPSEEK_AGENTIC",
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                "INVALID_DEEPSEEK_TOOL_RESPONSE",
                "DeepSeek returned an invalid tool response",
            ) from exc

        # Preserve provider-owned reasoning_content and tool_calls verbatim for
        # the next provider request. This dict is never persisted or emitted.
        provider_message: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
        }
        if "reasoning_content" in raw_message:
            provider_message["reasoning_content"] = raw_message["reasoning_content"]
        if "tool_calls" in raw_message:
            provider_message["tool_calls"] = raw_message["tool_calls"]
        return AssistantToolTurn(
            content=content,
            tool_calls=tuple(calls),
            provider_message=provider_message,
        )

    async def stream_answer(
        self,
        *,
        question: str,
        conversation_context: str,
        evidence: Sequence[Evidence],
        insufficient_answer: str | None = None,
    ) -> AsyncIterator[str]:
        facts = [
            {
                "evidence_id": item.evidence_id,
                "title": item.title,
                "headings": list(item.headings),
                "text": item.text,
                "source_location": _source_location(item),
                "derived_image_observations": [
                    observation.model_dump(mode="json")
                    for observation in item.image_observations
                ],
            }
            for item in evidence
        ]
        refusal_contract = (
            "If the evidence does not support an answer, return exactly this text "
            "with no citation marker:\n"
            f"{insufficient_answer}\n\n"
            if insufficient_answer
            else ""
        )
        user_content = (
            f"CURRENT QUESTION:\n{question}\n\n"
            "CONVERSATION CONTEXT (interpretation only; not evidence):\n"
            f"{conversation_context or '[none]'}\n\n"
            "OUTPUT CITATION CONTRACT:\n"
            "Cite every grounded claim with the exact marker "
            "[evidence:EVIDENCE_ID] on every visible prose line. Use only IDs "
            "present below. If no item "
            "supports an answer, state that the evidence is insufficient.\n\n"
            + refusal_contract
            + "RESOLVED FACT EVIDENCE:\n"
            + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
        )
        payload = {
            "model": self.answer_model,
            "messages": [
                {"role": "system", "content": self.answer_prompt},
                {"role": "user", "content": user_content},
            ],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
            "max_tokens": self.answer_max_tokens,
            "stream": True,
        }
        pending_whitespace = ""
        done_seen = False
        finish_reason: str | None = None
        try:
            async with (
                httpx.AsyncClient(
                    timeout=self.timeout, transport=self.transport
                ) as client,
                client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                ) as response,
            ):
                if response.is_error:
                    await response.aread()
                    raise ProviderError(
                        "DEEPSEEK_ANSWER_ERROR",
                        f"DeepSeek answer request failed with HTTP {response.status_code}",
                        retryable=response.status_code >= 500,
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        done_seen = True
                        break
                    try:
                        body = json.loads(data)
                        choice = body["choices"][0]
                        if finish_reason is not None:
                            raise ProviderError(
                                "INVALID_DEEPSEEK_STREAM",
                                "DeepSeek emitted data after the terminal choice",
                            )
                        delta = choice["delta"]
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise ProviderError(
                            "INVALID_DEEPSEEK_STREAM",
                            "DeepSeek returned an invalid answer stream",
                        ) from exc
                    except ProviderError:
                        raise
                    # reasoning_content is deliberately ignored and never leaves this method.
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(content, str) and content:
                        if not content.strip():
                            pending_whitespace += content
                        else:
                            yield pending_whitespace + content
                            pending_whitespace = ""
                    if choice.get("finish_reason") is not None:
                        finish_reason = _require_finish_reason(
                            choice,
                            allowed={"stop"},
                            code_prefix="DEEPSEEK_ANSWER",
                        )
                if not done_seen:
                    raise ProviderError(
                        "INVALID_DEEPSEEK_STREAM",
                        "DeepSeek answer stream ended without [DONE]",
                        retryable=True,
                    )
                if finish_reason != "stop":
                    raise ProviderError(
                        "INVALID_DEEPSEEK_STREAM",
                        "DeepSeek answer stream ended without finish_reason=stop",
                    )
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            raise ProviderError(
                "DEEPSEEK_ANSWER_ERROR",
                f"DeepSeek answer request failed: {exc}",
                retryable=True,
            ) from exc

    async def _post_json(self, payload: dict[str, Any], *, code: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self.transport
            ) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(
                code, f"DeepSeek request failed: {exc}", retryable=True
            ) from exc
        if not isinstance(body, dict):
            raise ProviderError(code, "DeepSeek response is not an object")
        return body


class ZhipuEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "embedding-3",
        dimensions: int = 2048,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not api_key:
            raise ProviderError("MISSING_ZHIPU_API_KEY", "ZHIPU_API_KEY is not set")
        if dimensions != 2048:
            raise ValueError("Zhipu Embedding-3 must use 2048 dimensions")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.transport = transport

    async def embed(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "input": [text],
            "dimensions": self.dimensions,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self.transport
            ) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
            vector = body["data"][0]["embedding"]
            if not isinstance(vector, list) or len(vector) != self.dimensions:
                raise ValueError("unexpected embedding dimensions")
            result = [float(item) for item in vector]
            if not all(math.isfinite(item) for item in result):
                raise ValueError("embedding contains a non-finite value")
            return result
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                "ZHIPU_EMBEDDING_ERROR",
                f"Zhipu embedding request failed: {exc}",
                retryable=True,
            ) from exc


class ZhipuVisionClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        prompt: str,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not api_key:
            raise ProviderError("MISSING_ZHIPU_API_KEY", "ZHIPU_API_KEY is not set")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt = prompt.strip()
        self.timeout = timeout
        self.transport = transport

    async def inspect(self, url: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self.transport
            ) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
            choice = body["choices"][0]
            _require_finish_reason(
                choice,
                allowed={"stop"},
                code_prefix="ZHIPU_VISION",
            )
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("vision response content is empty")
            return content.strip()
        except ProviderError:
            raise
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                "ZHIPU_VISION_ERROR",
                f"Zhipu vision request failed: {exc}",
                retryable=True,
            ) from exc


class ImageObservationCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            self._initialize(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=1.0)
        connection.execute("PRAGMA busy_timeout = 1000")
        return connection

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        schema = tuple(
            (str(row[1]), int(row[5]))
            for row in connection.execute(
                "PRAGMA table_info(image_observation_cache)"
            ).fetchall()
        )
        expected_schema = (
            ("image_sha256", 1),
            ("model", 2),
            ("prompt_sha256", 3),
            ("prompt_version", 4),
            ("observation", 0),
            ("created_at", 0),
        )
        if schema and schema != expected_schema:
            # Legacy rows are intentionally not copied: the old key did not bind
            # either image bytes or the actual prompt, so none are safe cache hits.
            connection.execute("DROP TABLE image_observation_cache")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_observation_cache (
                image_sha256 TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_sha256 TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                observation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(
                    image_sha256, model, prompt_sha256, prompt_version
                )
            ) STRICT
            """
        )

    def get(
        self,
        image_sha256: str,
        model: str,
        prompt_sha256: str,
        prompt_version: str,
    ) -> str | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT observation FROM image_observation_cache "
                "WHERE image_sha256 = ? AND model = ? "
                "AND prompt_sha256 = ? AND prompt_version = ?",
                (image_sha256, model, prompt_sha256, prompt_version),
            ).fetchone()
        return None if row is None else str(row[0])

    def put(
        self,
        image_sha256: str,
        model: str,
        prompt_sha256: str,
        prompt_version: str,
        observation: str,
    ) -> None:
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO image_observation_cache(
                    image_sha256, model, prompt_sha256, prompt_version,
                    observation, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    image_sha256, model, prompt_sha256, prompt_version
                ) DO UPDATE SET
                    observation=excluded.observation,
                    created_at=excluded.created_at
                """,
                (
                    image_sha256,
                    model,
                    prompt_sha256,
                    prompt_version,
                    observation,
                    timestamp,
                ),
            )


@dataclass(frozen=True)
class _SelectedImage:
    image_id: str
    url: str
    sha256: str | None
    caption: str | None

    @property
    def key(self) -> tuple[str, str | None]:
        return self.url, self.sha256


@dataclass(frozen=True)
class _ImageInspection:
    observation: str | None
    inspected: bool
    error: str | None = None
    vision_provider: str | None = None
    vision_model: str | None = None
    vision_prompt_sha256: str | None = None
    vision_prompt_version: str | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AssistantToolTurn:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    provider_message: dict[str, Any]


class ImageObserver:
    def __init__(
        self,
        client: ZhipuVisionClient,
        cache: ImageObservationCache,
        *,
        prompt_version: str = "visible-only-v1",
        max_images: int = 4,
        max_concurrency: int = 2,
    ):
        if max_images <= 0 or max_concurrency <= 0:
            raise ValueError("image observation budgets must be positive")
        self.client = client
        self.cache = cache
        self.prompt_version = prompt_version
        self.max_images = max_images
        self.max_concurrency = max_concurrency

    async def enrich(self, evidence: Sequence[Evidence]) -> tuple[Evidence, ...]:
        asserted_hashes_by_url: dict[str, set[str]] = {}
        reference_keys: list[tuple[str, str | None]] = []
        for item in evidence:
            for reference in _image_references(item):
                url = _normalize_remote_url(reference.url) if reference.url else None
                if not url:
                    continue
                reference_keys.append((url, reference.sha256))
                if reference.sha256 is not None:
                    asserted_hashes_by_url.setdefault(url, set()).add(reference.sha256)
        conflicted_urls = {
            url for url, hashes in asserted_hashes_by_url.items() if len(hashes) > 1
        }
        selected = self._select(evidence, excluded_urls=conflicted_urls)
        model = self.client.model
        prompt_sha256 = hashlib.sha256(self.client.prompt.encode("utf-8")).hexdigest()
        derivation = {
            "vision_provider": "zhipu",
            "vision_model": model,
            "vision_prompt_sha256": prompt_sha256,
            "vision_prompt_version": self.prompt_version,
        }
        inspections: dict[tuple[str, str | None], _ImageInspection] = {
            key: _ImageInspection(
                None,
                inspected=False,
                error="IMAGE_CONTENT_IDENTITY_CONFLICT",
                **derivation,
            )
            for key in reference_keys
            if key[0] in conflicted_urls
        }
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def inspect_image(
            image: _SelectedImage,
        ) -> tuple[tuple[str, str | None], _ImageInspection]:
            cached = None
            if image.sha256 is not None:
                try:
                    cached = await asyncio.to_thread(
                        self.cache.get,
                        image.sha256,
                        model,
                        prompt_sha256,
                        self.prompt_version,
                    )
                except sqlite3.Error:
                    # A cache lock/corruption must not prevent fresh inspection.
                    cached = None
            if cached is not None:
                return image.key, _ImageInspection(
                    cached, inspected=True, **derivation
                )
            try:
                async with semaphore:
                    observation = await self.client.inspect(image.url)
                if image.sha256 is not None:
                    try:
                        await asyncio.to_thread(
                            self.cache.put,
                            image.sha256,
                            model,
                            prompt_sha256,
                            self.prompt_version,
                            observation,
                        )
                    except sqlite3.Error:
                        # The observation remains useful even if cache persistence fails.
                        pass
                return image.key, _ImageInspection(
                    observation, inspected=True, **derivation
                )
            except ProviderError as exc:
                return image.key, _ImageInspection(
                    None, inspected=False, error=exc.code, **derivation
                )

        results = await asyncio.gather(*(inspect_image(image) for image in selected))
        inspections.update(results)
        return _attach_image_observations(evidence, inspections)

    def _select(
        self,
        evidence: Sequence[Evidence],
        *,
        excluded_urls: set[str] | None = None,
    ) -> tuple[_SelectedImage, ...]:
        return _select_images(
            evidence,
            max_images=self.max_images,
            excluded_urls=excluded_urls,
        )


def mark_images_uninspected(
    evidence: Sequence[Evidence],
    *,
    error: str,
    max_images: int = 4,
) -> tuple[Evidence, ...]:
    """Preserve ranked image URLs/captions when vision cannot inspect them."""
    selected = _select_images(evidence, max_images=max_images)
    inspections = {
        image.key: _ImageInspection(
            observation=None,
            inspected=False,
            error=error,
        )
        for image in selected
    }
    return _attach_image_observations(evidence, inspections)


def _select_images(
    evidence: Sequence[Evidence],
    *,
    max_images: int,
    excluded_urls: set[str] | None = None,
) -> tuple[_SelectedImage, ...]:
    result: list[_SelectedImage] = []
    seen: set[tuple[str, str | None]] = set()
    for item in evidence:
        for reference in _image_references(item):
            url = _normalize_remote_url(reference.url) if reference.url else None
            if not url or (excluded_urls is not None and url in excluded_urls):
                continue
            key = (url, reference.sha256)
            if key in seen:
                continue
            seen.add(key)
            result.append(
                _SelectedImage(
                    image_id=reference.image_id,
                    url=url,
                    sha256=reference.sha256,
                    caption=reference.caption,
                )
            )
            if len(result) == max_images:
                return tuple(result)
    return tuple(result)


def _attach_image_observations(
    evidence: Sequence[Evidence],
    inspections: dict[tuple[str, str | None], _ImageInspection],
) -> tuple[Evidence, ...]:
    enriched: list[Evidence] = []
    for item in evidence:
        derived: list[ImageObservation] = []
        attached: set[tuple[str, str | None]] = set()
        for reference in _image_references(item):
            url = _normalize_remote_url(reference.url) if reference.url else None
            if not url:
                continue
            key = (url, reference.sha256)
            if key in attached:
                continue
            inspection = inspections.get(key)
            if inspection is None:
                continue
            attached.add(key)
            derived.append(
                ImageObservation(
                    image_id=reference.image_id,
                    url=url,
                    image_sha256=reference.sha256,
                    caption=reference.caption,
                    observation=inspection.observation,
                    inspected=inspection.inspected,
                    error=inspection.error,
                    vision_provider=inspection.vision_provider,
                    vision_model=inspection.vision_model,
                    vision_prompt_sha256=inspection.vision_prompt_sha256,
                    vision_prompt_version=inspection.vision_prompt_version,
                )
            )
        if not derived:
            enriched.append(item)
            continue
        enriched.append(item.model_copy(update={"image_observations": tuple(derived)}))
    return tuple(enriched)


def _image_references(evidence: Evidence) -> tuple[Any, ...]:
    locator = evidence.locator
    return locator.images if isinstance(locator, DocumentChunkLocator) else ()


def _source_location(evidence: Evidence) -> dict[str, Any]:
    locator = evidence.locator
    if isinstance(locator, DocumentChunkLocator):
        return {
            "document_id": locator.document_id,
            "chunk_ordinal": locator.ordinal,
            "segment_ordinal": locator.segment_ordinal,
            "raw_path": locator.raw_path,
            "enhanced_path": locator.enhanced_path,
            "docling_path": locator.docling_path,
            "fact_document_path": locator.fact_document_path,
            "self_refs": list(locator.self_refs),
            "doc_item_refs": list(evidence.doc_item_refs),
        }
    return {
        "project_id": getattr(locator, "project_id", None),
        "path": getattr(locator, "path", None),
    }


def _completion_content(body: dict[str, Any], *, code_prefix: str) -> str:
    """Read one complete JSON-mode response without accepting truncation."""

    try:
        choice = body["choices"][0]
        _require_finish_reason(choice, allowed={"stop"}, code_prefix=code_prefix)
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("completion content is not a string")
        return content
    except ProviderError:
        raise
    except (IndexError, KeyError, TypeError) as exc:
        raise ProviderError(
            f"{code_prefix}_INVALID_RESPONSE",
            "Provider returned an invalid JSON completion",
        ) from exc


def _stage2_evidence(evidence: Sequence[Evidence]) -> list[dict[str, Any]]:
    """Serialize only resolved, immutable evidence fields for Stage 2 calls."""

    return [
        {
            "evidence_id": item.evidence_id,
            "title": item.title,
            "headings": list(item.headings),
            "text": item.text,
            "source_location": _source_location(item),
            "derived_image_observations": [
                observation.model_dump(mode="json")
                for observation in item.image_observations
            ],
        }
        for item in evidence
    ]


def _normalize_remote_url(value: str) -> str | None:
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return None
    if parts.scheme.casefold() not in {"http", "https"} or not parts.hostname:
        return None
    return urlunsplit(
        (parts.scheme.casefold(), parts.netloc.casefold(), parts.path, parts.query, "")
    )


def _unique_nonempty(values: Sequence[str], *, limit: int) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = " ".join(raw.split())
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
        if len(result) == limit:
            break
    return tuple(result)
