from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
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


def _image_evidence(*, url: str, sha256: str | None) -> Evidence:
    return Evidence(
        evidence_id="source:doc:chunk",
        evidence_kind="source_chunk",
        locator=_locator((ImageReference(image_id="chart", url=url, sha256=sha256),)),
        text="Fact",
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
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "semantic_query_en": "How to improve retention",
                                        "lexical_terms_en": ["retention", "cohort"],
                                        "lexical_phrases_en": ["cohort retention"],
                                    }
                                )
                            },
                        }
                    ]
                },
            )
        stream = (
            'data: {"choices":[{"delta":{"reasoning_content":"PRIVATE_CHAIN"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"Visible"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, content=stream, headers={"content-type": "text/event-stream"}
        )

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
                insufficient_answer="现有证据不足，无法可靠回答这个问题。",
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
    assert requests[0]["max_tokens"] == 1_024
    assert requests[1]["thinking"] == {"type": "enabled"}
    assert requests[1]["reasoning_effort"] == "max"
    assert requests[1]["max_tokens"] == 64_000
    assert "temperature" not in requests[1]
    assert "top_p" not in requests[1]
    answer_input = requests[1]["messages"][1]["content"]
    assert "The visible retention curve rises." in answer_input
    assert "https://cdn.example.com/chart.png" in answer_input
    assert "现有证据不足，无法可靠回答这个问题。" in answer_input


def test_deepseek_answer_stream_requires_done_terminal() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )

    async def consume() -> None:
        async for _ in _deepseek(transport).stream_answer(
            question="Question",
            conversation_context="",
            evidence=(),
        ):
            pass

    with pytest.raises(ProviderError) as caught:
        asyncio.run(consume())
    assert caught.value.code == "INVALID_DEEPSEEK_STREAM"


def test_deepseek_answer_rejects_length_truncation_before_done() -> None:
    stream = (
        'data: {"choices":[{"delta":{"content":"[evidence:source:doc:chunk] Partial"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
        "data: [DONE]\n\n"
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            content=stream,
            headers={"content-type": "text/event-stream"},
        )
    )

    async def consume() -> None:
        async for _ in _deepseek(transport).stream_answer(
            question="Question",
            conversation_context="",
            evidence=(),
        ):
            pass

    with pytest.raises(ProviderError) as caught:
        asyncio.run(consume())
    assert caught.value.code == "DEEPSEEK_ANSWER_TRUNCATED"


def test_chinese_rewrite_failure_is_lossless_safe_fallback() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    rewrite = asyncio.run(
        _deepseek(transport).rewrite("Notion 在 2024 年如何增长？", "")
    )

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
                        "finish_reason": "tool_calls",
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
                        },
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
    assert captured[0]["max_tokens"] == 32_000
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


def test_deepseek_tool_turn_rejects_length_truncation() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "[evidence:source:doc:chunk] Partial"},
                    }
                ]
            },
        )
    )

    with pytest.raises(ProviderError) as caught:
        asyncio.run(
            _deepseek(transport).tool_turn(
                messages=[{"role": "user", "content": "Question"}],
                tools=[],
            )
        )
    assert caught.value.code == "DEEPSEEK_AGENTIC_TRUNCATED"


def test_vision_rejects_length_truncation() -> None:
    vision = ZhipuVisionClient(
        api_key="test-key",
        base_url="https://zhipu.test/v4",
        model="glm-5v-turbo",
        prompt="Visible only.",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "Partial description"},
                        }
                    ]
                },
            )
        ),
    )

    with pytest.raises(ProviderError) as caught:
        asyncio.run(vision.inspect("https://cdn.example.com/chart.png"))
    assert caught.value.code == "ZHIPU_VISION_TRUNCATED"


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


def test_image_observer_bounds_parallel_vision_calls(tmp_path: Path) -> None:
    active = 0
    peak = 0

    class SlowVision:
        model = "vision-test"
        prompt = "Visible only."

        async def inspect(self, url: str) -> str:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                return f"Observed {url}"
            finally:
                active -= 1

    evidence = Evidence(
        evidence_id="source:doc:parallel",
        evidence_kind="source_chunk",
        locator=_locator(
            tuple(
                ImageReference(
                    image_id=f"parallel-{index}",
                    url=f"https://cdn.example.com/parallel-{index}.png",
                    sha256=f"{index + 1:064x}",
                )
                for index in range(4)
            )
        ),
        text="Fact",
    )
    observer = ImageObserver(
        SlowVision(),  # type: ignore[arg-type]
        ImageObservationCache(tmp_path / "cache.sqlite3"),
        max_concurrency=2,
    )

    enriched = asyncio.run(observer.enrich((evidence,)))

    assert peak == 2
    assert len(enriched[0].image_observations) == 4
    assert all(item.inspected for item in enriched[0].image_observations)


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
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "A visible chart."},
                    }
                ]
            },
        )

    evidence = _image_evidence(url="https://cdn.example.com/chart.png", sha256="1" * 64)
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
    observation = first[0].image_observations[0]
    assert observation.observation == "A visible chart."
    assert observation.image_sha256 == "1" * 64
    assert observation.vision_provider == "zhipu"
    assert observation.vision_model == "glm-5v-turbo"
    assert (
        observation.vision_prompt_sha256 == hashlib.sha256(b"Visible only.").hexdigest()
    )
    assert observation.vision_prompt_version == "visible-only-v1"
    assert second[0].image_observations[0].observation == "A visible chart."


def test_vision_cache_same_url_with_changed_content_hash_is_a_miss(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": f"Observation {calls}."},
                    }
                ]
            },
        )

    url = "https://cdn.example.com/chart.png"
    vision = ZhipuVisionClient(
        api_key="test-key",
        base_url="https://zhipu.test/v4",
        model="glm-5v-turbo",
        prompt="Visible only.",
        transport=httpx.MockTransport(handler),
    )
    observer = ImageObserver(vision, ImageObservationCache(tmp_path / "cache.sqlite3"))

    first = asyncio.run(observer.enrich((_image_evidence(url=url, sha256="1" * 64),)))
    changed = asyncio.run(observer.enrich((_image_evidence(url=url, sha256="2" * 64),)))
    first_again = asyncio.run(
        observer.enrich((_image_evidence(url=url, sha256="1" * 64),))
    )

    assert calls == 2
    assert first[0].image_observations[0].observation == "Observation 1."
    assert changed[0].image_observations[0].observation == "Observation 2."
    assert first_again[0].image_observations[0].observation == "Observation 1."


def test_vision_same_url_preserves_each_evidence_reference_identity(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Shared pixels."},
                    }
                ]
            },
        )

    url = "https://cdn.example.com/shared.png"
    sha256 = "3" * 64
    first_reference = ImageReference(
        image_id="doc-a#image-1",
        url=url,
        sha256=sha256,
        caption="Caption from document A",
    )
    second_reference = ImageReference(
        image_id="doc-b#image-2",
        url=url,
        sha256=sha256,
        caption="Caption from document B",
    )
    first = Evidence(
        evidence_id="source:doc-a:chunk",
        evidence_kind="source_chunk",
        locator=_locator((first_reference,)),
        text="Fact A",
    )
    second = Evidence(
        evidence_id="source:doc-b:chunk",
        evidence_kind="source_chunk",
        locator=_locator((second_reference,)),
        text="Fact B",
    )
    observer = ImageObserver(
        ZhipuVisionClient(
            api_key="test-key",
            base_url="https://zhipu.test/v4",
            model="glm-5v-turbo",
            prompt="Visible only.",
            transport=httpx.MockTransport(handler),
        ),
        ImageObservationCache(tmp_path / "cache.sqlite3"),
    )

    enriched = asyncio.run(observer.enrich((first, second)))

    assert calls == 1
    first_observation = enriched[0].image_observations[0]
    second_observation = enriched[1].image_observations[0]
    assert (first_observation.image_id, first_observation.caption) == (
        first_reference.image_id,
        first_reference.caption,
    )
    assert (second_observation.image_id, second_observation.caption) == (
        second_reference.image_id,
        second_reference.caption,
    )
    assert first_observation.observation == second_observation.observation


def test_vision_same_url_with_conflicting_asserted_hashes_fails_closed(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Should not be used."},
                    }
                ]
            },
        )

    url = "https://cdn.example.com/mutable.png"
    first = _image_evidence(url=url, sha256="1" * 64)
    second = _image_evidence(url=url, sha256="2" * 64).model_copy(
        update={"evidence_id": "source:doc:chunk-two"}
    )
    observer = ImageObserver(
        ZhipuVisionClient(
            api_key="test-key",
            base_url="https://zhipu.test/v4",
            model="glm-5v-turbo",
            prompt="Visible only.",
            transport=httpx.MockTransport(handler),
        ),
        ImageObservationCache(tmp_path / "cache.sqlite3"),
    )

    enriched = asyncio.run(observer.enrich((first, second)))

    assert calls == 0
    assert [item.image_observations[0].inspected for item in enriched] == [
        False,
        False,
    ]
    assert [item.image_observations[0].error for item in enriched] == [
        "IMAGE_CONTENT_IDENTITY_CONFLICT",
        "IMAGE_CONTENT_IDENTITY_CONFLICT",
    ]


def test_vision_detects_hash_conflict_beyond_inspection_limit(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Visible."},
                    }
                ]
            },
        )

    shared_url = "https://cdn.example.com/shared-mutable.png"
    specs = (
        (shared_url, "1" * 64),
        ("https://cdn.example.com/b.png", "b" * 64),
        ("https://cdn.example.com/c.png", "c" * 64),
        ("https://cdn.example.com/d.png", "d" * 64),
        (shared_url, "2" * 64),
    )
    evidence = tuple(
        _image_evidence(url=url, sha256=sha256).model_copy(
            update={"evidence_id": f"source:doc:chunk-{index}"}
        )
        for index, (url, sha256) in enumerate(specs)
    )
    observer = ImageObserver(
        ZhipuVisionClient(
            api_key="test-key",
            base_url="https://zhipu.test/v4",
            model="glm-5v-turbo",
            prompt="Visible only.",
            transport=httpx.MockTransport(handler),
        ),
        ImageObservationCache(tmp_path / "cache.sqlite3"),
        max_images=4,
    )

    enriched = asyncio.run(observer.enrich(evidence))

    assert calls == 3
    assert enriched[0].image_observations[0].error == (
        "IMAGE_CONTENT_IDENTITY_CONFLICT"
    )
    assert enriched[4].image_observations[0].error == (
        "IMAGE_CONTENT_IDENTITY_CONFLICT"
    )
    assert all(enriched[index].image_observations[0].inspected for index in (1, 2, 3))


def test_vision_cache_actual_prompt_change_is_a_miss(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        prompt = payload["messages"][0]["content"][1]["text"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": f"For {prompt}"},
                    }
                ]
            },
        )

    cache = ImageObservationCache(tmp_path / "cache.sqlite3")
    transport = httpx.MockTransport(handler)
    vision = ZhipuVisionClient(
        api_key="test-key",
        base_url="https://zhipu.test/v4",
        model="glm-5v-turbo",
        prompt="First prompt.",
        transport=transport,
    )
    observer = ImageObserver(vision, cache)
    evidence = _image_evidence(url="https://cdn.example.com/chart.png", sha256="1" * 64)

    first = asyncio.run(observer.enrich((evidence,)))
    vision.prompt = "Changed prompt."
    changed = asyncio.run(observer.enrich((evidence,)))
    vision.prompt = "First prompt."
    first_again = asyncio.run(observer.enrich((evidence,)))

    assert calls == 2
    assert first[0].image_observations[0].observation == "For First prompt."
    assert changed[0].image_observations[0].observation == "For Changed prompt."
    assert first_again[0].image_observations[0].observation == "For First prompt."


def test_vision_image_without_hash_is_not_cached_across_runs(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Fresh observation."},
                    }
                ]
            },
        )

    vision = ZhipuVisionClient(
        api_key="test-key",
        base_url="https://zhipu.test/v4",
        model="glm-5v-turbo",
        prompt="Visible only.",
        transport=httpx.MockTransport(handler),
    )
    observer = ImageObserver(vision, ImageObservationCache(tmp_path / "cache.sqlite3"))
    evidence = _image_evidence(url="https://cdn.example.com/chart.png", sha256=None)

    asyncio.run(observer.enrich((evidence,)))
    asyncio.run(observer.enrich((evidence,)))

    assert calls == 2


def test_legacy_vision_cache_rows_are_not_reused(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.sqlite3"
    connection = sqlite3.connect(cache_path)
    try:
        connection.execute(
            """
            CREATE TABLE image_observation_cache (
                url TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                observation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(url, model, prompt_version)
            ) STRICT
            """
        )
        connection.execute(
            "INSERT INTO image_observation_cache VALUES (?, ?, ?, ?, ?)",
            (
                "https://cdn.example.com/chart.png",
                "glm-5v-turbo",
                "visible-only-v1",
                "Unverifiable legacy observation.",
                "2026-01-01T00:00:00Z",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Fresh observation."},
                    }
                ]
            },
        )

    observer = ImageObserver(
        ZhipuVisionClient(
            api_key="test-key",
            base_url="https://zhipu.test/v4",
            model="glm-5v-turbo",
            prompt="Visible only.",
            transport=httpx.MockTransport(handler),
        ),
        ImageObservationCache(cache_path),
    )
    evidence = _image_evidence(url="https://cdn.example.com/chart.png", sha256="1" * 64)

    first = asyncio.run(observer.enrich((evidence,)))
    second = asyncio.run(observer.enrich((evidence,)))

    assert calls == 1
    assert first[0].image_observations[0].observation == "Fresh observation."
    assert second[0].image_observations[0].observation == "Fresh observation."


def test_vision_cache_io_runs_off_the_event_loop_thread(tmp_path: Path) -> None:
    cache_threads: list[int] = []

    class RecordingCache(ImageObservationCache):
        def get(
            self,
            image_sha256: str,
            model: str,
            prompt_sha256: str,
            prompt_version: str,
        ) -> str | None:
            cache_threads.append(threading.get_ident())
            return super().get(image_sha256, model, prompt_sha256, prompt_version)

        def put(
            self,
            image_sha256: str,
            model: str,
            prompt_sha256: str,
            prompt_version: str,
            observation: str,
        ) -> None:
            cache_threads.append(threading.get_ident())
            super().put(
                image_sha256,
                model,
                prompt_sha256,
                prompt_version,
                observation,
            )

    vision = ZhipuVisionClient(
        api_key="test-key",
        base_url="https://zhipu.test/v4",
        model="glm-5v-turbo",
        prompt="Visible only.",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "Observation."},
                        }
                    ]
                },
            )
        ),
    )
    observer = ImageObserver(vision, RecordingCache(tmp_path / "cache.sqlite3"))
    evidence = _image_evidence(url="https://cdn.example.com/chart.png", sha256="1" * 64)
    event_loop_thread = threading.get_ident()

    asyncio.run(observer.enrich((evidence,)))
    asyncio.run(observer.enrich((evidence,)))

    assert len(cache_threads) == 3
    assert all(thread_id != event_loop_thread for thread_id in cache_threads)


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
