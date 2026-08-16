from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

from .errors import RetrievalError

EMBEDDING_DIMENSIONS = 2048
MAX_EMBEDDING_BATCH = 64
ZHIPU_INPUT_TOO_LONG_CODE = "1261"


class EmbeddingError(RetrievalError):
    pass


class EmbeddingInputTooLongError(EmbeddingError):
    """Provider-confirmed input-length rejection eligible for safe splitting."""

    def __init__(self, message: str = "embedding input exceeds the provider limit"):
        super().__init__("EMBEDDING_INPUT_TOO_LONG", message)


class EmbeddingProvider(Protocol):
    model_name: str
    dimensions: int
    status: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_all(self, texts: Sequence[str]) -> list[list[float]]: ...


def validate_embeddings(
    embeddings: Sequence[Sequence[float]],
    *,
    expected_count: int,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> list[list[float]]:
    if len(embeddings) != expected_count:
        raise EmbeddingError(
            "EMBEDDING_COUNT_MISMATCH",
            f"expected {expected_count} embeddings, received {len(embeddings)}",
        )
    validated: list[list[float]] = []
    for index, embedding in enumerate(embeddings):
        if len(embedding) != dimensions:
            raise EmbeddingError(
                "EMBEDDING_DIMENSION_MISMATCH",
                f"embedding {index} has {len(embedding)} dimensions; expected {dimensions}",
            )
        vector: list[float] = []
        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingError(
                    "INVALID_EMBEDDING_VALUE",
                    f"embedding {index} contains a non-numeric value",
                )
            number = float(value)
            if not math.isfinite(number):
                raise EmbeddingError(
                    "INVALID_EMBEDDING_VALUE",
                    f"embedding {index} contains a non-finite value",
                )
            vector.append(number)
        validated.append(vector)
    return validated


class ZhipuEmbeddingClient:
    """Strict synchronous client for the domestic Zhipu Embedding-3 API."""

    model_name = "embedding-3"
    dimensions = EMBEDDING_DIMENSIONS
    status = "ready"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.api_key = api_key or os.environ.get("ZHIPU_API_KEY")
        if not self.api_key:
            raise EmbeddingError("MISSING_ZHIPU_API_KEY", "ZHIPU_API_KEY is not set")
        self.base_url = (
            base_url
            or os.environ.get("ZHIPU_BASE_URL")
            or "https://open.bigmodel.cn/api/paas/v4"
        ).rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > MAX_EMBEDDING_BATCH:
            raise EmbeddingError(
                "EMBEDDING_BATCH_TOO_LARGE",
                f"Embedding-3 accepts at most {MAX_EMBEDDING_BATCH} inputs per request",
            )
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError(
                "INVALID_EMBEDDING_INPUT", "embedding inputs must be non-empty strings"
            )

        payload = {
            "model": self.model_name,
            "input": list(texts),
            "dimensions": self.dimensions,
        }
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                "EMBEDDING_API_ERROR", f"Zhipu embedding request failed: {exc}"
            ) from exc

        try:
            body = response.json()
        except ValueError as exc:
            if response.is_error:
                try:
                    response.raise_for_status()
                except httpx.HTTPError as http_error:
                    raise EmbeddingError(
                        "EMBEDDING_API_ERROR",
                        f"Zhipu embedding request failed: {http_error}",
                    ) from http_error
            raise EmbeddingError(
                "INVALID_EMBEDDING_RESPONSE", "response is not valid JSON"
            ) from exc

        if response.is_error:
            error_code = _zhipu_error_code(body)
            if response.status_code == 400 and error_code == ZHIPU_INPUT_TOO_LONG_CODE:
                raise EmbeddingInputTooLongError(
                    "Zhipu rejected embedding input with business error code 1261"
                )
            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    "EMBEDDING_API_ERROR",
                    f"Zhipu embedding request failed with business code "
                    f"{error_code or 'unknown'}: {exc}",
                ) from exc

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise EmbeddingError(
                "INVALID_EMBEDDING_RESPONSE", "response.data is not an array"
            )
        try:
            ordered = sorted(data, key=lambda item: int(item["index"]))
            vectors = [item["embedding"] for item in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(
                "INVALID_EMBEDDING_RESPONSE",
                "response items must contain numeric index and embedding",
            ) from exc
        if [int(item["index"]) for item in ordered] != list(range(len(texts))):
            raise EmbeddingError(
                "INVALID_EMBEDDING_RESPONSE",
                "response embedding indexes are incomplete or duplicated",
            )
        return validate_embeddings(
            vectors, expected_count=len(texts), dimensions=self.dimensions
        )

    def embed_all(self, texts: Sequence[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for start in range(0, len(texts), MAX_EMBEDDING_BATCH):
            result.extend(self.embed(texts[start : start + MAX_EMBEDDING_BATCH]))
        return result


class MockEmbeddingProvider:
    """Deterministic test-only embedder; never reported as production-ready."""

    model_name = "mock-hash-embedding"
    dimensions = EMBEDDING_DIMENSIONS
    status = "test_only"
    _token_pattern = re.compile(r"[\w+#.-]+", re.UNICODE)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if len(texts) > MAX_EMBEDDING_BATCH:
            raise EmbeddingError(
                "EMBEDDING_BATCH_TOO_LARGE",
                f"mock embedder accepts at most {MAX_EMBEDDING_BATCH} inputs per request",
            )
        vectors = [self._embed_one(text) for text in texts]
        return validate_embeddings(
            vectors, expected_count=len(texts), dimensions=self.dimensions
        )

    def embed_all(self, texts: Sequence[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for start in range(0, len(texts), MAX_EMBEDDING_BATCH):
            result.extend(self.embed(texts[start : start + MAX_EMBEDDING_BATCH]))
        return result

    @classmethod
    def _embed_one(cls, text: str) -> list[float]:
        vector = [0.0] * cls.dimensions
        tokens = cls._token_pattern.findall(text.casefold())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "little") % cls.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[slot] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "little") % cls.dimensions] = 1.0
            return vector
        return [value / norm for value in vector]


def _zhipu_error_code(body: object) -> str | None:
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    value = error.get("code") if isinstance(error, dict) else body.get("code")
    return None if value is None else str(value)
