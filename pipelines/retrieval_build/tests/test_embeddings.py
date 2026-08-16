from __future__ import annotations

import json

import httpx
import pytest
from retrieval_build.embeddings import (
    EMBEDDING_DIMENSIONS,
    EmbeddingError,
    EmbeddingInputTooLongError,
    MockEmbeddingProvider,
    ZhipuEmbeddingClient,
    validate_embeddings,
)


def test_zhipu_request_explicitly_sets_model_dimensions_and_batch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "embedding-3"
        assert payload["dimensions"] == 2048
        assert len(payload["input"]) == 2
        data = [
            {"index": 1, "embedding": [0.0] * EMBEDDING_DIMENSIONS},
            {"index": 0, "embedding": [1.0] * EMBEDDING_DIMENSIONS},
        ]
        return httpx.Response(200, json={"data": data})

    client = ZhipuEmbeddingClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    result = client.embed(["first", "second"])
    assert result[0][0] == 1.0
    assert result[1][0] == 0.0


def test_zhipu_rejects_batches_over_64_before_network() -> None:
    client = ZhipuEmbeddingClient(
        api_key="test-key", transport=httpx.MockTransport(lambda _: None)
    )
    with pytest.raises(EmbeddingError) as caught:
        client.embed(["x"] * 65)
    assert caught.value.code == "EMBEDDING_BATCH_TOO_LARGE"


def test_embedding_dimension_and_finite_values_are_strict() -> None:
    with pytest.raises(EmbeddingError) as dimension_error:
        validate_embeddings([[0.0] * 10], expected_count=1)
    assert dimension_error.value.code == "EMBEDDING_DIMENSION_MISMATCH"

    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[17] = float("nan")
    with pytest.raises(EmbeddingError) as finite_error:
        validate_embeddings([vector], expected_count=1)
    assert finite_error.value.code == "INVALID_EMBEDDING_VALUE"


def test_mock_embedder_is_explicitly_test_only_and_2048_dimensions() -> None:
    embedder = MockEmbeddingProvider()
    result = embedder.embed(["same text", "same text"])
    assert embedder.status == "test_only"
    assert len(result[0]) == EMBEDDING_DIMENSIONS
    assert result[0] == result[1]


def test_zhipu_maps_only_official_400_business_code_1261_to_too_long() -> None:
    def too_long(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"code": "1261", "message": "Prompt too long"}},
        )

    client = ZhipuEmbeddingClient(
        api_key="test-key", transport=httpx.MockTransport(too_long)
    )
    with pytest.raises(EmbeddingInputTooLongError) as caught:
        client.embed(["long input"])
    assert caught.value.code == "EMBEDDING_INPUT_TOO_LONG"

    def other_bad_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"code": "1214", "message": "input is too long"}},
        )

    client = ZhipuEmbeddingClient(
        api_key="test-key", transport=httpx.MockTransport(other_bad_request)
    )
    with pytest.raises(EmbeddingError) as generic:
        client.embed(["bad input"])
    assert not isinstance(generic.value, EmbeddingInputTooLongError)
    assert generic.value.code == "EMBEDDING_API_ERROR"
