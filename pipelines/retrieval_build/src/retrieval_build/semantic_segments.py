from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeAlias

from .embeddings import EmbeddingInputTooLongError, EmbeddingProvider

_MAX_SPLIT_DEPTH = 64
_BOUNDARY_PATTERNS = (
    re.compile(r"\n[ \t]*\n"),
    re.compile(r"\n"),
    re.compile(r"(?<=[.!?。！？;；])\s+"),
    re.compile(r"\s+"),
)
_Pending: TypeAlias = tuple[int, str, int]


@dataclass(frozen=True, slots=True)
class EmbeddedTextSegment:
    text: str
    embedding: tuple[float, ...]


def embed_with_length_fallback(
    provider: EmbeddingProvider,
    texts: list[str],
) -> tuple[tuple[EmbeddedTextSegment, ...], ...]:
    """Embed normally, splitting only provider-confirmed overlong inputs.

    Results retain input order and left-to-right segment order. Any exception
    other than ``EmbeddingInputTooLongError`` propagates unchanged.
    """

    if not texts:
        return ()
    pending: list[_Pending] = [(index, text, 0) for index, text in enumerate(texts)]
    embedded = _embed_group(provider, pending)
    buckets: list[list[EmbeddedTextSegment]] = [[] for _ in texts]
    for parent_index, segment in embedded:
        buckets[parent_index].append(segment)
    return tuple(tuple(bucket) for bucket in buckets)


def split_text_deterministically(text: str) -> tuple[str, str]:
    """Bisect near the midpoint, preferring structural/text boundaries."""

    normalized = text.strip()
    if len(normalized) < 2:
        raise EmbeddingInputTooLongError(
            "provider rejected an input that cannot be split further"
        )
    midpoint = len(normalized) // 2
    for pattern in _BOUNDARY_PATTERNS:
        candidates: list[tuple[int, int]] = []
        for match in pattern.finditer(normalized):
            boundary = match.end()
            left = normalized[:boundary].strip()
            right = normalized[boundary:].strip()
            if left and right:
                candidates.append((abs(boundary - midpoint), boundary))
        if candidates:
            _, boundary = min(candidates)
            return normalized[:boundary].strip(), normalized[boundary:].strip()

    left = normalized[:midpoint].strip()
    right = normalized[midpoint:].strip()
    if not left or not right:
        raise EmbeddingInputTooLongError(
            "provider rejected an input that cannot be split further"
        )
    return left, right


def _embed_group(
    provider: EmbeddingProvider,
    items: list[_Pending],
) -> list[tuple[int, EmbeddedTextSegment]]:
    try:
        embeddings = provider.embed_all([text for _, text, _ in items])
    except EmbeddingInputTooLongError:
        if len(items) > 1:
            midpoint = len(items) // 2
            return _embed_group(provider, items[:midpoint]) + _embed_group(
                provider, items[midpoint:]
            )

        parent_index, text, depth = items[0]
        if depth >= _MAX_SPLIT_DEPTH:
            raise EmbeddingInputTooLongError(
                f"input {parent_index} remains too long after {_MAX_SPLIT_DEPTH} splits"
            )
        left, right = split_text_deterministically(text)
        return _embed_group(provider, [(parent_index, left, depth + 1)]) + _embed_group(
            provider, [(parent_index, right, depth + 1)]
        )

    if len(embeddings) != len(items):
        # Providers are expected to enforce this themselves, but fail closed if
        # an alternate implementation violates the shared contract.
        raise ValueError("embedding provider returned an unexpected result count")
    return [
        (
            parent_index,
            EmbeddedTextSegment(text=text, embedding=tuple(embedding)),
        )
        for (parent_index, text, _), embedding in zip(items, embeddings, strict=True)
    ]
