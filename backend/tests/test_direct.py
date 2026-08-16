from __future__ import annotations

import asyncio
import json

import httpx
from ai_pm_contracts import (
    AnswerRequest,
    DocumentChunkLocator,
    DocumentSegment,
    Evidence,
    ImageReference,
    RetrievalHit,
)
from lenny_backend.conversations import Message
from lenny_backend.direct import (
    DirectAnswerService,
    SearchOutcome,
    build_recent_context,
    conservative_token_estimate,
)
from lenny_backend.providers import DeepSeekClient


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


class _Retrieval:
    def __init__(self, locator: DocumentChunkLocator):
        self.locator = locator

    async def search(self, rewrite):
        assert rewrite.semantic_query_en == "SECRET_REWRITE"
        return SearchOutcome(
            keyword_hits=(
                RetrievalHit(
                    provider="keyword",
                    rank=1,
                    native_score=999.123,
                    locator=self.locator,
                ),
            ),
            vector_hits=(),
            errors=("SEMANTIC_PROVIDER_UNAVAILABLE",),
        )


class _Facts:
    def __init__(self, evidence: Evidence):
        self.evidence = evidence

    def read_source_window(self, locator, window=0):
        assert locator == self.evidence.locator
        assert window == 0
        return (self.evidence,)


def test_direct_service_bm25_degraded_and_fact_prompt_barrier() -> None:
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
                                        "semantic_query_en": "SECRET_REWRITE",
                                        "lexical_terms_en": ["retention"],
                                        "lexical_phrases_en": [],
                                    }
                                )
                            }
                        }
                    ]
                },
            )
        stream = (
            'data: {"choices":[{"delta":{"reasoning_content":"DO_NOT_LEAK"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"Supported answer"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=stream)

    locator = _locator()
    evidence = Evidence(
        evidence_id="source:doc:chunk",
        evidence_kind="source_chunk",
        locator=locator,
        text="ONLY_RESOLVED_FACT_TEXT",
        title="Fact title",
    )
    deepseek = DeepSeekClient(
        api_key="test-key",
        base_url="https://deepseek.test/v1",
        rewrite_model="deepseek-v4-flash",
        answer_model="deepseek-v4-flash",
        rewrite_prompt="Rewrite as JSON.",
        answer_prompt="Use facts only.",
        transport=httpx.MockTransport(handler),
    )
    service = DirectAnswerService(
        deepseek=deepseek,
        retrieval=_Retrieval(locator),
        facts=_Facts(evidence),
    )
    history = (
        Message(
            id="message-one",
            conversation_id="conversation-one",
            role="user",
            content="What works?",
            created_at="2026-01-01T00:00:00Z",
        ),
    )

    async def collect():
        return [
            event
            async for event in service.stream_answer(
                conversation_id="conversation-one",
                request=AnswerRequest(question="What works?", mode="direct"),
                history=history,
            )
        ]

    events = asyncio.run(collect())
    types = [event.type for event in events]
    assert types == [
        "phase",
        "phase",
        "retrieval_summary",
        "phase",
        "evidence",
        "phase",
        "answer_delta",
        "done",
    ]
    summary = next(event for event in events if event.type == "retrieval_summary")
    assert summary.provider_counts == {"keyword": 1, "vector": 0}
    assert summary.fused_count == 1
    answer_request = requests[1]
    answer_prompt = answer_request["messages"][1]["content"]  # type: ignore[index]
    assert "ONLY_RESOLVED_FACT_TEXT" in answer_prompt
    assert '"source_location"' in answer_prompt
    assert '"document_id":"doc"' in answer_prompt
    assert '"raw_path":"raw/newsletter/doc.md"' in answer_prompt
    assert '"self_refs":["#/texts/0"]' in answer_prompt
    assert "SECRET_REWRITE" not in answer_prompt
    assert "999.123" not in answer_prompt
    serialized_events = "".join(event.model_dump_json() for event in events)
    assert "DO_NOT_LEAK" not in serialized_events


def test_recent_context_uses_conservative_token_budget_for_chinese() -> None:
    history = (
        Message(
            id="old",
            conversation_id="conversation",
            role="assistant",
            content="旧" * 20_000,
            created_at="2026-01-01T00:00:00Z",
        ),
        Message(
            id="current",
            conversation_id="conversation",
            role="user",
            content="当前问题",
            created_at="2026-01-01T00:01:00Z",
        ),
    )
    context = build_recent_context(history, current_question="当前问题")
    assert conservative_token_estimate(context) <= 24_000
    assert len(context) < 20_000


class _BrokenImageObserver:
    async def enrich(self, evidence):
        raise OSError("cache unavailable")


def test_direct_preserves_image_caption_and_url_when_vision_is_unavailable() -> None:
    for image_observer, expected_error in (
        (None, "ZHIPU_VISION_NOT_CONFIGURED"),
        (_BrokenImageObserver(), "ZHIPU_VISION_UNAVAILABLE"),
    ):
        requests: list[dict[str, object]] = []

        def handler(
            request: httpx.Request, requests: list[dict[str, object]] = requests
        ) -> httpx.Response:
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
                                            "semantic_query_en": "SECRET_REWRITE",
                                            "lexical_terms_en": ["retention"],
                                            "lexical_phrases_en": [],
                                        }
                                    )
                                }
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                content=(
                    'data: {"choices":[{"delta":{"content":"Answer"}}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        image = ImageReference(
            image_id="image-one",
            url="https://cdn.example.com/chart.png",
            caption="Retention chart",
        )
        locator = _locator((image,))
        evidence = Evidence(
            evidence_id="source:doc:chunk",
            evidence_kind="source_chunk",
            locator=locator,
            text="Resolved fact with an image",
        )
        deepseek = DeepSeekClient(
            api_key="test-key",
            base_url="https://deepseek.test/v1",
            rewrite_model="deepseek-v4-flash",
            answer_model="deepseek-v4-flash",
            rewrite_prompt="Rewrite as JSON.",
            answer_prompt="Use facts only.",
            transport=httpx.MockTransport(handler),
        )
        service = DirectAnswerService(
            deepseek=deepseek,
            retrieval=_Retrieval(locator),
            facts=_Facts(evidence),
            image_observer=image_observer,  # type: ignore[arg-type]
        )

        async def collect(service: DirectAnswerService = service):
            return [
                event
                async for event in service.stream_answer(
                    conversation_id="conversation-one",
                    request=AnswerRequest(question="What does the chart show?", mode="direct"),
                    history=(),
                )
            ]

        events = asyncio.run(collect())
        assert any(
            event.type == "phase" and event.phase == "inspect_images"
            for event in events
        )
        emitted = next(event.evidence for event in events if event.type == "evidence")
        observation = emitted.image_observations[0]
        assert observation.inspected is False
        assert observation.error == expected_error
        assert observation.url == image.url
        assert observation.caption == image.caption
        answer_prompt = requests[1]["messages"][1]["content"]  # type: ignore[index]
        assert image.url in answer_prompt
        assert image.caption in answer_prompt
        assert expected_error in answer_prompt


def test_recent_context_keeps_at_most_twelve_rounds() -> None:
    history = tuple(
        Message(
            id=f"message-{index}",
            conversation_id="conversation",
            role="user" if index % 2 == 0 else "assistant",
            content=f"content-{index}",
            created_at=f"2026-01-01T00:00:{index:02d}Z",
        )
        for index in range(30)
    )
    context = build_recent_context(history, current_question="not-the-last-message")
    assert len(context.splitlines()) == 24
    assert "content-5" not in context
    assert "content-6" in context
