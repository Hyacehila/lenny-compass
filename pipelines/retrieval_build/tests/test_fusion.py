from __future__ import annotations

from ai_pm_contracts import LlmWikiPageLocator, RetrievalHit
from conftest import make_document
from retrieval_build.fusion import reciprocal_rank_fusion


def test_rrf_rewards_overlap_and_keeps_component_ranks() -> None:
    document = make_document("rrf", ["a", "b", "c"])
    a, b, c = [chunk.locator for chunk in document.chunks]
    keyword = (
        RetrievalHit(provider="keyword", rank=1, native_score=-3.0, locator=a),
        RetrievalHit(provider="keyword", rank=2, native_score=-2.0, locator=b),
    )
    vector = (
        RetrievalHit(provider="vector", rank=1, native_score=0.1, locator=b),
        RetrievalHit(provider="vector", rank=2, native_score=0.2, locator=c),
    )

    fused = reciprocal_rank_fusion((keyword, vector), k=60, limit=12)

    assert fused[0].locator.chunk_id == b.chunk_id
    assert fused[0].component_ranks == {"keyword": 2, "vector": 1}
    assert fused[0].rrf_score == (1 / 62) + (1 / 61)
    assert len(fused) == 3


def test_rrf_is_deterministic_and_caps_at_twelve() -> None:
    document = make_document("many", [str(index) for index in range(20)])
    ranking = tuple(
        RetrievalHit(
            provider="keyword",
            rank=index,
            native_score=-float(index),
            locator=chunk.locator,
        )
        for index, chunk in enumerate(document.chunks, start=1)
    )
    first = reciprocal_rank_fusion((ranking,))
    second = reciprocal_rank_fusion((ranking,))
    assert first == second
    assert len(first) == 12


def test_rrf_uses_stable_llm_wiki_project_and_path_key() -> None:
    locator = LlmWikiPageLocator(project_id="ai-pm", path="growth/activation")
    ranking = (
        RetrievalHit(provider="llm_wiki", rank=1, native_score=0.9, locator=locator),
    )
    fused = reciprocal_rank_fusion((ranking,))
    assert fused[0].locator == locator


def test_rrf_deduplicates_multiple_embedding_segments_of_one_parent_chunk() -> None:
    locator = make_document("segments", ["parent chunk"]).chunks[0].locator
    vector = (
        RetrievalHit(
            provider="vector",
            rank=1,
            native_score=0.01,
            locator=locator,
            retrieval_segment_ordinal=0,
        ),
        RetrievalHit(
            provider="vector",
            rank=2,
            native_score=0.02,
            locator=locator,
            retrieval_segment_ordinal=1,
        ),
    )
    fused = reciprocal_rank_fusion((vector,))
    assert len(fused) == 1
    assert fused[0].component_ranks == {"vector": 1}
