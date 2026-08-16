from __future__ import annotations

import pytest
from conftest import ThresholdEmbeddingProvider
from retrieval_build.embeddings import EmbeddingError, EmbeddingInputTooLongError
from retrieval_build.semantic_segments import (
    embed_with_length_fallback,
    split_text_deterministically,
)


def test_confirmed_overlong_input_splits_deterministically_at_boundaries() -> None:
    text = (
        "Product teams should define activation clearly.\n\n"
        "Then they should validate the signal with retained users.\n\n"
        "Finally they should revisit the metric as the product evolves."
    )
    first_provider = ThresholdEmbeddingProvider(maximum_characters=55)
    second_provider = ThresholdEmbeddingProvider(maximum_characters=55)

    first = embed_with_length_fallback(first_provider, ["short input", text])
    second = embed_with_length_fallback(second_provider, ["short input", text])

    assert len(first[0]) == 1
    assert len(first[1]) > 1
    assert [segment.text for segment in first[1]] == [
        segment.text for segment in second[1]
    ]
    assert all(len(segment.text) <= 55 for segment in first[1])


def test_non_length_provider_errors_fail_closed_without_splitting() -> None:
    class FailingProvider(ThresholdEmbeddingProvider):
        def embed(self, texts):
            self.calls.append(tuple(texts))
            raise EmbeddingError("EMBEDDING_API_ERROR", "authentication failed")

    provider = FailingProvider(maximum_characters=1)
    with pytest.raises(EmbeddingError) as caught:
        embed_with_length_fallback(provider, ["this must not be split"])
    assert caught.value.code == "EMBEDDING_API_ERROR"
    assert provider.calls == [("this must not be split",)]


def test_unsplittable_provider_confirmed_input_remains_a_hard_failure() -> None:
    provider = ThresholdEmbeddingProvider(maximum_characters=0)
    with pytest.raises(EmbeddingInputTooLongError):
        embed_with_length_fallback(provider, ["x"])


def test_boundary_split_prefers_paragraph_nearest_midpoint() -> None:
    assert split_text_deterministically("alpha beta\n\ngamma delta") == (
        "alpha beta",
        "gamma delta",
    )
