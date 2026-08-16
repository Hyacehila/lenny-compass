from __future__ import annotations

from collections.abc import Iterable, Sequence

from ai_pm_contracts import (
    DocumentChunkLocator,
    FusedRetrievalHit,
    LlmWikiPageLocator,
    RetrievalHit,
)


def _locator_key(locator: DocumentChunkLocator | LlmWikiPageLocator) -> str:
    if isinstance(locator, DocumentChunkLocator):
        return (
            f"document:{locator.document_id}:{locator.chunk_id}:"
            f"{locator.raw_sha256}:{locator.enhanced_sha256}:"
            f"{locator.chunk_sha256}"
        )
    return (
        f"wiki:{locator.project_id}:{locator.path}:"
        f"{locator.revision or ''}:{locator.fact_id or ''}"
    )


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[RetrievalHit]],
    *,
    k: int = 60,
    limit: int = 12,
) -> tuple[FusedRetrievalHit, ...]:
    """Fuse equal-weight ranked lists using 1/(k + one-based rank)."""

    if isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    accumulator: dict[str, dict[str, object]] = {}
    for ranking in rankings:
        seen_in_provider: set[tuple[str, str]] = set()
        for hit in ranking:
            provider = str(hit.provider)
            key = _locator_key(hit.locator)
            provider_key = (provider, key)
            if provider_key in seen_in_provider:
                continue
            seen_in_provider.add(provider_key)
            item = accumulator.setdefault(
                key,
                {
                    "locator": hit.locator,
                    "score": 0.0,
                    "ranks": {},
                    "scores": {},
                },
            )
            ranks = item["ranks"]
            scores = item["scores"]
            assert isinstance(ranks, dict)
            assert isinstance(scores, dict)
            ranks[provider] = hit.rank
            scores[provider] = hit.native_score
            item["score"] = float(item["score"]) + (1.0 / (k + hit.rank))

    ordered = sorted(
        accumulator.items(),
        key=lambda pair: (
            -float(pair[1]["score"]),
            min(int(rank) for rank in dict(pair[1]["ranks"]).values()),
            pair[0],
        ),
    )[:limit]

    return tuple(
        FusedRetrievalHit(
            rank=index,
            rrf_score=float(item["score"]),
            locator=item["locator"],
            component_ranks=dict(item["ranks"]),
            component_scores=dict(item["scores"]),
        )
        for index, (_, item) in enumerate(ordered, start=1)
    )
