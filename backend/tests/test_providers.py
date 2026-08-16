from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from ai_pm_contracts import (
    DocumentChunkLocator,
    DocumentSegment,
    Evidence,
    ImageObservation,
    ImageReference,
)
from lenny_backend.providers import (
    DeepSeekClient,
    ImageObservationCache,
    ImageObserver,
    ProviderError,
    ZhipuEmbeddingClient,
    ZhipuVisionClient,
)


def _locator(images: tuple[ImageReference, ...] = ()) -> DocumentChunkLocator:
    return DocumentChunkLocator(
        document_id="doc",
        chunk_id="chunk",
        ordinal=0,
        segment_ordinal=0,
        fact_document_path="documents/v1/doc.json",
        raw_path="raw/newsletter/doc.md",
        raw_sha256="a" * 64,
        enhanced_path="enhanced/newsletter/doc.md",
        enhanced_sha256="c" * 64,
        docling_path="documents/v1/doc.docling.json",
        docling_sha256="d" * 64,
        chunk_sha256="b" * 64,
        segment=DocumentSegment(
            first_self_ref="#/texts/0",
            last_self_ref="#/texts/0",
            item_count=1,
        ),
        self_refs=("#/texts/0",),
        images=images,
    )


def _deepseek(transport: httpx.MockTransport) -> DeepSeekClient:
    return DeepSeekClient(
        api_key="test-key",
        base_url="https://deepseek.test/v1",
        rewrite_model="deepseek-v4-flash",
        answer_model="deepseek-v4-flash",
        rewrite_prompt="Return the required JSON object.",
        answer_prompt="Use only resolved fact evidence.",
        transport=transport,
    )


def test_deepseek_payloads_and_reasoning_is_never_yielded() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if payload["stream"] is False:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "semantic_query_en": "How to improve retention",
                                        "lexical_terms_en": ["retention", "cohort"],
                                        "lexical_phrases_en": ["cohort retention"],
                                    }
                                )
                            }
                        }
                    ]
                },
            )
        stream = (
            'data: {"choices":[{"delta":{"reasoning_content":"PRIVATE_CHAIN"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"Visible"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

    async def exercise() -> tuple[object, list[str]]:
        client = _deepseek(httpx.MockTransport(handler))
        rewrite = await client.rewrite("如何提高留存？", "")
        image = ImageReference(
            image_id="chart",
            url="https://cdn.example.com/chart.png",
            caption="Retention chart",
        )
        evidence = Evidence(
            evidence_id="source:doc:chunk",
            evidence_kind="source_chunk",
            locator=_locator((image,)),
            text="Fact with a chart",
            image_observations=(
                ImageObservation(
                    image_id=image.image_id,
                    url=image.url,
                    caption=image.caption,
                    observation="The visible retention curve rises.",
                    inspected=True,
                ),
            ),
        )
        deltas = [
            item
            async for item in client.stream_answer(
                question="如何提高留存？",
                conversation_context="",
                evidence=(evidence,),
            )
        ]
        return rewrite, deltas

    rewrite, deltas = asyncio.run(exercise())
    assert rewrite.semantic_query_en == "How to improve retention"
    assert deltas == ["Visible", " answer"]
    assert "PRIVATE_CHAIN" not in "".join(deltas)
    assert requests[0]["thinking"] == {"type": "enabled"}
    assert requests[0]["reasoning_effort"] == "max"
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert requests[1]["thinking"] == {"type": "enabled"}
    assert requests[1]["reasoning_effort"] == "max"
    assert "temperature" not in requests[1]
    assert "top_p" not in requests[1]
    answer_input = requests[1]["messages"][1]["content"]
    assert "The visible retention curve rises." in answer_input
    assert "https://cdn.example.com/chart.png" in answer_input


def test_chinese_rewrite_failure_is_lossless_safe_fallback() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    rewrite = asyncio.run(_deepseek(transport).rewrite("Notion 在 2024 年如何增长？", ""))

    assert rewrite.used_fallback is True
    assert rewrite.original_query == "Notion 在 2024 年如何增长？"
    assert rewrite.semantic_query_en == rewrite.original_query
    assert "Notion" in rewrite.lexical_terms_en
    assert "2024" in rewrite.lexical_terms_en


def test_deepseek_thinking_tool_turn_uses_compatible_payload_and_replay() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "PRIVATE_TOOL_REASONING",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_keyword",
                                        "arguments": '{"query":"retention"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    turn = asyncio.run(
        _deepseek(httpx.MockTransport(handler)).tool_turn(
            messages=[{"role": "user", "content": "Find retention guidance"}],
            tools=[{"type": "function", "function": {"name": "search_keyword"}}],
        )
    )

    assert "tool_choice" not in captured[0]
    assert captured[0]["thinking"] == {"type": "enabled"}
    assert captured[0]["reasoning_effort"] == "max"
    assert turn.provider_message == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "PRIVATE_TOOL_REASONING",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "search_keyword",
                    "arguments": '{"query":"retention"}',
                },
            }
        ],
    }
    assert turn.tool_calls[0].name == "search_keyword"


def test_vision_failure_returns_typed_uninspected_markers_and_deduplicates(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    images = tuple(
        ImageReference(
            image_id=f"image-{index}",
            url=f"https://cdn.example.com/{index}.png#fragment",
            caption=f"Caption {index}",
        )
        for index in range(5)
    ) + (
        ImageReference(
            image_id="duplicate",
            url="https://cdn.example.com/0.png",
            caption="duplicate",
        ),
    )
    evidence = Evidence(
        evidence_id="source:doc:chunk",
        evidence_kind="source_chunk",
        locator=_locator(images),
        text="Fact",
    )
    vision = ZhipuVisionClient(
        api_key="test-key",
        base_url="https://zhipu.test/v4",
        model="glm-5v-turbo",
        prompt="Describe visible facts only.",
        transport=httpx.MockTransport(handler),
    )
    observer = ImageObserver(vision, ImageObservationCache(tmp_path / "cache.sqlite3"))

    enriched = asyncio.run(observer.enrich((evidence,)))
    observations = enriched[0].image_observations
    assert calls == 4
    assert len(observations) == 4
    assert all(item.inspected is False for item in observations)
    assert all(item.error == "ZHIPU_VISION_ERROR" for item in observations)
    assert observations[0].url == "https://cdn.example.com/0.png"
    assert observations[0].caption == "Caption 0"


def test_successful_vision_observation_is_cached(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        image_input = payload["messages"][0]["content"][0]
        assert image_input == {
            "type": "image_url",
            "image_url": {"url": "https://cdn.example.com/chart.png"},
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "A visible chart."}}]},
        )

    evidence = Evidence(
        evidence_id="source:doc:chunk",
        evidence_kind="source_chunk",
        locator=_locator(
            (
                ImageReference(
                    image_id="chart", url="https://cdn.example.com/chart.png"
                ),
            )
        ),
        text="Fact",
    )
    vision = ZhipuVisionClient(
        api_key="test-key",
        base_url="https://zhipu.test/v4",
        model="glm-5v-turbo",
        prompt="Visible only.",
        transport=httpx.MockTransport(handler),
    )
    observer = ImageObserver(vision, ImageObservationCache(tmp_path / "cache.sqlite3"))

    first = asyncio.run(observer.enrich((evidence,)))
    second = asyncio.run(observer.enrich((evidence,)))
    assert calls == 1
    assert first[0].image_observations[0].observation == "A visible chart."
    assert second[0].image_observations[0].observation == "A visible chart."


def test_embedding_rejects_non_finite_2048d_response() -> None:
    vector = [0.0] * 2048
    vector[17] = float("nan")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=json.dumps({"data": [{"index": 0, "embedding": vector}]}),
        )
    )
    client = ZhipuEmbeddingClient(
        api_key="test-key",
        base_url="https://zhipu.test/v4",
        transport=transport,
    )
    with pytest.raises(ProviderError, match="non-finite"):
        asyncio.run(client.embed("query"))
