from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import AsyncIterator, Sequence
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
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not api_key:
            raise ProviderError("MISSING_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY is not set")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.rewrite_model = rewrite_model
        self.answer_model = answer_model
        self.rewrite_prompt = rewrite_prompt.strip()
        self.answer_prompt = answer_prompt.strip()
        self.timeout = timeout
        self.transport = transport

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
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        try:
            body = await self._post_json(payload, code="DEEPSEEK_REWRITE_ERROR")
            content = body["choices"][0]["message"]["content"]
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
            "stream": False,
        }
        body = await self._post_json(payload, code="DEEPSEEK_AGENTIC_ERROR")
        try:
            raw_message = body["choices"][0]["message"]
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
        user_content = (
            f"CURRENT QUESTION:\n{question}\n\n"
            "CONVERSATION CONTEXT (interpretation only; not evidence):\n"
            f"{conversation_context or '[none]'}\n\n"
            "RESOLVED FACT EVIDENCE:\n"
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
            "stream": True,
        }
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
                    if not data or data == "[DONE]":
                        continue
                    try:
                        body = json.loads(data)
                        delta = body["choices"][0]["delta"]
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ProviderError(
                            "INVALID_DEEPSEEK_STREAM",
                            "DeepSeek returned an invalid answer stream",
                        ) from exc
                    # reasoning_content is deliberately ignored and never leaves this method.
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(content, str) and content:
                        yield content
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
            raise ProviderError(code, f"DeepSeek request failed: {exc}", retryable=True) from exc
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
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("vision response content is empty")
            return content.strip()
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
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_observation_cache (
                    url TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    observation TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(url, model, prompt_version)
                ) STRICT
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def get(self, url: str, model: str, prompt_version: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT observation FROM image_observation_cache "
                "WHERE url = ? AND model = ? AND prompt_version = ?",
                (url, model, prompt_version),
            ).fetchone()
        return None if row is None else str(row[0])

    def put(self, url: str, model: str, prompt_version: str, observation: str) -> None:
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO image_observation_cache(
                    url, model, prompt_version, observation, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(url, model, prompt_version) DO UPDATE SET
                    observation=excluded.observation,
                    created_at=excluded.created_at
                """,
                (url, model, prompt_version, observation, timestamp),
            )


@dataclass(frozen=True)
class _SelectedImage:
    image_id: str
    url: str
    caption: str | None


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
    ):
        self.client = client
        self.cache = cache
        self.prompt_version = prompt_version
        self.max_images = max_images

    async def enrich(self, evidence: Sequence[Evidence]) -> tuple[Evidence, ...]:
        selected = self._select(evidence)
        observations: dict[str, ImageObservation] = {}
        for image in selected:
            cached = self.cache.get(image.url, self.client.model, self.prompt_version)
            if cached is not None:
                observations[image.url] = self._record(image, cached, inspected=True)
                continue
            try:
                observation = await self.client.inspect(image.url)
                self.cache.put(
                    image.url, self.client.model, self.prompt_version, observation
                )
                observations[image.url] = self._record(
                    image, observation, inspected=True
                )
            except ProviderError as exc:
                observations[image.url] = self._record(
                    image, None, inspected=False, error=exc.code
                )

        enriched: list[Evidence] = []
        for item in evidence:
            urls = {
                normalized
                for reference in _image_references(item)
                if reference.url and (normalized := _normalize_remote_url(reference.url))
            }
            derived = [observations[url] for url in observations if url in urls]
            if not derived:
                enriched.append(item)
                continue
            enriched.append(item.model_copy(update={"image_observations": tuple(derived)}))
        return tuple(enriched)

    def _select(self, evidence: Sequence[Evidence]) -> tuple[_SelectedImage, ...]:
        return _select_images(evidence, max_images=self.max_images)

    def _record(
        self,
        image: _SelectedImage,
        observation: str | None,
        *,
        inspected: bool,
        error: str | None = None,
    ) -> ImageObservation:
        return ImageObservation(
            image_id=image.image_id,
            url=image.url,
            caption=image.caption,
            observation=observation,
            inspected=inspected,
            error=error,
        )


def mark_images_uninspected(
    evidence: Sequence[Evidence],
    *,
    error: str,
    max_images: int = 4,
) -> tuple[Evidence, ...]:
    """Preserve ranked image URLs/captions when vision cannot inspect them."""
    selected = _select_images(evidence, max_images=max_images)
    observations = {
        image.url: ImageObservation(
            image_id=image.image_id,
            url=image.url,
            caption=image.caption,
            observation=None,
            inspected=False,
            error=error,
        )
        for image in selected
    }
    return _attach_image_observations(evidence, observations)


def _select_images(
    evidence: Sequence[Evidence], *, max_images: int
) -> tuple[_SelectedImage, ...]:
    result: list[_SelectedImage] = []
    seen: set[str] = set()
    for item in evidence:
        for reference in _image_references(item):
            url = _normalize_remote_url(reference.url) if reference.url else None
            if not url or url in seen:
                continue
            seen.add(url)
            result.append(
                _SelectedImage(
                    image_id=reference.image_id,
                    url=url,
                    caption=reference.caption,
                )
            )
            if len(result) == max_images:
                return tuple(result)
    return tuple(result)


def _attach_image_observations(
    evidence: Sequence[Evidence], observations: dict[str, ImageObservation]
) -> tuple[Evidence, ...]:
    enriched: list[Evidence] = []
    for item in evidence:
        urls = {
            normalized
            for reference in _image_references(item)
            if reference.url and (normalized := _normalize_remote_url(reference.url))
        }
        derived = [observations[url] for url in observations if url in urls]
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
