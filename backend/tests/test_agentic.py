from __future__ import annotations

import asyncio
import json
from collections import OrderedDict

import httpx
import pytest
from ai_pm_contracts import (
    AnswerRequest,
    DocumentChunkLocator,
    DocumentSegment,
    Evidence,
    ImageReference,
    RetrievalHit,
)
from lenny_backend.agentic import AGENTIC_MAX_READS, AGENTIC_TOOLS, AgenticAnswerService
from lenny_backend.conversations import Message
from lenny_backend.grounding import insufficient_evidence_answer
from lenny_backend.providers import (
    AssistantToolTurn,
    DeepSeekClient,
    ProviderError,
    ToolCall,
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


def _evidence(locator: DocumentChunkLocator, text: str = "READ_FACT") -> Evidence:
    return Evidence(
        evidence_id="source:doc:chunk",
        evidence_kind="source_chunk",
        locator=locator,
        text=text,
    )


class _Retrieval:
    def __init__(self, locator: DocumentChunkLocator):
        self.locator = locator
        self.calls: list[tuple[str, str, int]] = []

    async def keyword(self, query: str, *, limit: int = 50):
        self.calls.append(("keyword", query, limit))
        return (
            RetrievalHit(
                provider="keyword",
                rank=1,
                native_score=12345.0,
                locator=self.locator,
            ),
        )

    async def semantic(self, query: str, *, limit: int = 50):
        self.calls.append(("vector", query, limit))
        return ()


class _Facts:
    def __init__(self, evidence: Evidence):
        self.evidence = evidence

    def read_source_window(self, locator, window=0):
        return (self.evidence,)


class _UnavailableRetrieval:
    async def keyword(self, query: str, *, limit: int = 50):
        raise OSError("lexical unavailable")

    async def semantic(self, query: str, *, limit: int = 50):
        raise OSError("semantic unavailable")


def _history() -> tuple[Message, ...]:
    return (
        Message(
            id="message",
            conversation_id="conversation",
            role="user",
            content="Question",
            created_at="2026-01-01T00:00:00Z",
        ),
    )


def test_agentic_reasoning_roundtrip_fact_gate_and_sse_secrecy() -> None:
    locator = _locator()
    evidence = _evidence(locator)
    requests: list[dict[str, object]] = []
    private_reasoning = "原样 PRIVATE reasoning\nline two"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            message = {
                "role": "assistant",
                "content": None,
                "reasoning_content": private_reasoning,
                "tool_calls": [
                    {
                        "id": "search-1",
                        "type": "function",
                        "function": {
                            "name": "search_keyword",
                            "arguments": json.dumps({"query": "retention", "limit": 3}),
                        },
                    }
                ],
            }
        elif turn == 2:
            assistant = payload["messages"][-2]
            assert assistant["reasoning_content"] == private_reasoning
            search_result = json.loads(payload["messages"][-1]["content"])
            assert set(search_result["hits"][0]) == {"provider", "rank", "locator"}
            assert "text" not in search_result["hits"][0]
            message = {
                "role": "assistant",
                "content": None,
                "reasoning_content": "SECOND_PRIVATE",
                "tool_calls": [
                    {
                        "id": "read-1",
                        "type": "function",
                        "function": {
                            "name": "read_source_window",
                            "arguments": json.dumps(
                                {
                                    "locator": locator.model_dump(mode="json"),
                                    "window": 0,
                                }
                            ),
                        },
                    }
                ],
            }
        else:
            assistant = payload["messages"][-2]
            assert assistant["reasoning_content"] == "SECOND_PRIVATE"
            read_result = json.loads(payload["messages"][-1]["content"])
            assert read_result["evidence"][0]["text"] == "READ_FACT"
            message = {
                "role": "assistant",
                "content": "[evidence:source:doc:chunk] Supported answer",
                "reasoning_content": "FINAL_PRIVATE",
            }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": (
                            "tool_calls" if "tool_calls" in message else "stop"
                        ),
                        "message": message,
                    }
                ]
            },
        )

    deepseek = DeepSeekClient(
        api_key="test-key",
        base_url="https://deepseek.test/v1",
        rewrite_model="deepseek-v4-flash",
        answer_model="deepseek-v4-flash",
        rewrite_prompt="unused",
        answer_prompt="unused",
        transport=httpx.MockTransport(handler),
    )
    retrieval = _Retrieval(locator)
    service = AgenticAnswerService(
        deepseek=deepseek,
        retrieval=retrieval,  # type: ignore[arg-type]
        facts=_Facts(evidence),
        system_prompt="Read before citing.",
    )

    async def collect():
        return [
            event
            async for event in service.stream_answer(
                conversation_id="conversation",
                request=AnswerRequest(question="Question", mode="agentic"),
                history=_history(),
            )
        ]

    events = asyncio.run(collect())
    assert [event.type for event in events] == [
        "phase",
        "phase",
        "retrieval_summary",
        "evidence",
        "phase",
        "answer_delta",
        "done",
    ]
    done = events[-1]
    assert done.response.evidence == (evidence,)  # type: ignore[union-attr]
    serialized = "".join(event.model_dump_json() for event in events)
    assert "PRIVATE" not in serialized
    assert all(request["thinking"] == {"type": "enabled"} for request in requests)
    assert all(request["reasoning_effort"] == "max" for request in requests)
    initial_prompt = requests[0]["messages"][1]["content"]
    assert "INITIAL RETRIEVAL HITS" in initial_prompt
    assert "12345.0" not in initial_prompt
    assert retrieval.calls[:2] == [
        ("keyword", "Question", 6),
        ("vector", "Question", 6),
    ]
    assert retrieval.calls[2] == ("keyword", "retention", 3)
    summary = next(event for event in events if event.type == "retrieval_summary")
    assert summary.provider_counts == {"keyword": 2, "vector": 0}
    assert summary.fused_count == 0


class _FinalModel:
    def __init__(self, answer: str):
        self.answer = answer

    async def tool_turn(self, *, messages, tools):
        return AssistantToolTurn(
            content=self.answer,
            tool_calls=(),
            provider_message={
                "role": "assistant",
                "content": self.answer,
                "reasoning_content": "never emitted",
            },
        )


def test_unread_evidence_citation_is_rejected() -> None:
    locator = _locator()
    service = AgenticAnswerService(
        deepseek=_FinalModel("[evidence:not-read] unsupported"),
        retrieval=_Retrieval(locator),  # type: ignore[arg-type]
        facts=_Facts(_evidence(locator)),
        system_prompt="Read first.",
    )

    async def consume() -> None:
        async for _ in service.stream_answer(
            conversation_id="conversation",
            request=AnswerRequest(question="Question", mode="agentic"),
            history=_history(),
        ):
            pass

    with pytest.raises(ProviderError) as caught:
        asyncio.run(consume())
    assert caught.value.code == "AGENTIC_UNREAD_CITATION"


def test_agentic_does_not_misreport_total_retrieval_failure_as_no_evidence() -> None:
    service = AgenticAnswerService(
        deepseek=_FinalModel("No evidence"),
        retrieval=_UnavailableRetrieval(),  # type: ignore[arg-type]
        facts=_Facts(_evidence(_locator())),
        system_prompt="Read first.",
    )

    async def consume() -> None:
        async for _ in service.stream_answer(
            conversation_id="conversation",
            request=AnswerRequest(question="Question", mode="agentic"),
            history=(),
        ):
            pass

    with pytest.raises(ProviderError) as caught:
        asyncio.run(consume())
    assert caught.value.code == "RETRIEVAL_UNAVAILABLE"


class _PartiallyLeakyRetrieval:
    async def keyword(self, query: str, *, limit: int = 50):
        raise OSError(r"private local path D:\secret\index.sqlite3")

    async def semantic(self, query: str, *, limit: int = 50):
        return ()


class _InitialPromptCaptureModel:
    initial_prompt = ""

    async def tool_turn(self, *, messages, tools):
        self.initial_prompt = messages[-1]["content"]
        return AssistantToolTurn(
            content="No evidence",
            tool_calls=(),
            provider_message={"role": "assistant", "content": "No evidence"},
        )


def test_agentic_initial_search_error_does_not_disclose_local_exception() -> None:
    model = _InitialPromptCaptureModel()
    service = AgenticAnswerService(
        deepseek=model,
        retrieval=_PartiallyLeakyRetrieval(),  # type: ignore[arg-type]
        facts=_Facts(_evidence(_locator())),
        system_prompt="Read first.",
    )

    async def consume() -> None:
        async for _ in service.stream_answer(
            conversation_id="conversation",
            request=AnswerRequest(question="Question", mode="agentic"),
            history=(),
        ):
            pass

    asyncio.run(consume())
    assert "D:\\secret" not in model.initial_prompt
    assert "Local search could not complete the request" in model.initial_prompt


class _ReadThenFinishModel:
    def __init__(
        self, locator: DocumentChunkLocator, final_answer: str = "No evidence"
    ):
        self.locator = locator
        self.final_answer = final_answer
        self.calls = 0
        self.tool_result = ""

    async def tool_turn(self, *, messages, tools):
        self.calls += 1
        if self.calls == 1:
            arguments = json.dumps(
                {"locator": self.locator.model_dump(mode="json"), "window": 0}
            )
            raw = {
                "id": "read-secret",
                "type": "function",
                "function": {
                    "name": "read_source_window",
                    "arguments": arguments,
                },
            }
            return AssistantToolTurn(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="read-secret",
                        name="read_source_window",
                        arguments=arguments,
                    ),
                ),
                provider_message={"role": "assistant", "tool_calls": [raw]},
            )
        self.tool_result = messages[-1]["content"]
        return AssistantToolTurn(
            content=self.final_answer,
            tool_calls=(),
            provider_message={"role": "assistant", "content": self.final_answer},
        )


class _LeakyFactFailure(RuntimeError):
    def __init__(self, code: str):
        super().__init__(r"fact missing at D:\private\facts\document.json")
        self.code = code


class _LeakyFacts:
    def __init__(self, code: str):
        self.code = code

    def read_source_window(self, locator, window=0):
        raise _LeakyFactFailure(self.code)


@pytest.mark.parametrize(
    ("exception_code", "public_code"),
    [
        (r"SECRET_D:\\private", "FACT_READ_ERROR"),
    ],
)
def test_agentic_fact_failure_is_sanitized_and_not_misreported_as_insufficient(
    exception_code: str, public_code: str
) -> None:
    locator = _locator()
    model = _ReadThenFinishModel(locator)
    service = AgenticAnswerService(
        deepseek=model,
        retrieval=_Retrieval(locator),  # type: ignore[arg-type]
        facts=_LeakyFacts(exception_code),
        system_prompt="Read first.",
    )

    async def consume() -> None:
        async for _ in service.stream_answer(
            conversation_id="conversation",
            request=AnswerRequest(question="Question", mode="agentic"),
            history=(),
        ):
            pass

    with pytest.raises(ProviderError) as caught:
        asyncio.run(consume())
    assert caught.value.code == "FACT_RESOLUTION_FAILED"
    assert "D:\\private" not in model.tool_result
    assert json.loads(model.tool_result)["error"] == {
        "code": public_code,
        "message": "Source evidence could not be read from the validated snapshot",
    }


class _ReadTwiceModel:
    def __init__(self, locator: DocumentChunkLocator):
        self.locator = locator
        self.calls = 0
        self.first_tool_result = ""

    async def tool_turn(self, *, messages, tools):
        self.calls += 1
        if self.calls == 2:
            self.first_tool_result = messages[-1]["content"]
        arguments = json.dumps(
            {"locator": self.locator.model_dump(mode="json"), "window": 0}
        )
        raw = {
            "id": f"read-{self.calls}",
            "type": "function",
            "function": {
                "name": "read_source_window",
                "arguments": arguments,
            },
        }
        return AssistantToolTurn(
            content=None,
            tool_calls=(
                ToolCall(
                    id=f"read-{self.calls}",
                    name="read_source_window",
                    arguments=arguments,
                ),
            ),
            provider_message={"role": "assistant", "tool_calls": [raw]},
        )


class _EvidenceThenSnapshotChange:
    def __init__(self, evidence: Evidence):
        self.evidence = evidence
        self.calls = 0

    def read_source_window(self, locator, window=0):
        self.calls += 1
        if self.calls == 1:
            return (self.evidence,)
        raise _LeakyFactFailure("RETRIEVAL_SNAPSHOT_CHANGED")


def test_agentic_snapshot_change_is_fatal_after_evidence_was_read() -> None:
    locator = _locator()
    model = _ReadTwiceModel(locator)
    facts = _EvidenceThenSnapshotChange(_evidence(locator))
    service = AgenticAnswerService(
        deepseek=model,
        retrieval=_Retrieval(locator),  # type: ignore[arg-type]
        facts=facts,
        system_prompt="Read first.",
    )

    async def consume() -> None:
        async for _ in service.stream_answer(
            conversation_id="conversation",
            request=AnswerRequest(question="Question", mode="agentic"),
            history=(),
        ):
            pass

    with pytest.raises(ProviderError) as caught:
        asyncio.run(consume())
    assert caught.value.code == "RETRIEVAL_SNAPSHOT_CHANGED"
    assert facts.calls == 2
    assert "READ_FACT" in model.first_tool_result


def test_agentic_can_refuse_after_reading_irrelevant_evidence() -> None:
    locator = _locator()
    refusal = insufficient_evidence_answer("Question")
    service = AgenticAnswerService(
        deepseek=_ReadThenFinishModel(locator, refusal),
        retrieval=_Retrieval(locator),  # type: ignore[arg-type]
        facts=_Facts(_evidence(locator)),
        system_prompt="Read first.",
    )

    async def collect():
        return [
            event
            async for event in service.stream_answer(
                conversation_id="conversation",
                request=AnswerRequest(question="Question", mode="agentic"),
                history=(),
            )
        ]

    done = asyncio.run(collect())[-1]
    assert done.type == "done"
    assert done.response.answer_kind == "insufficient_evidence"  # type: ignore[union-attr]
    assert done.response.evidence == ()  # type: ignore[union-attr]
    assert done.response.claims == ()  # type: ignore[union-attr]


def test_agentic_provider_messages_have_a_cumulative_budget() -> None:
    locator = _locator()
    service = AgenticAnswerService(
        deepseek=_FinalModel("No evidence"),
        retrieval=_Retrieval(locator),  # type: ignore[arg-type]
        facts=_Facts(_evidence(locator)),
        system_prompt="Read first.",
        max_provider_output_tokens=1,
    )

    async def consume() -> None:
        async for _ in service.stream_answer(
            conversation_id="conversation",
            request=AnswerRequest(question="Question", mode="agentic"),
            history=(),
        ):
            pass

    with pytest.raises(ProviderError) as caught:
        asyncio.run(consume())
    assert caught.value.code == "AGENTIC_PROVIDER_MESSAGE_LIMIT"


def test_agentic_tool_output_overflow_rolls_back_unseen_evidence() -> None:
    locator = _locator()
    model = _ReadThenFinishModel(locator, insufficient_evidence_answer("Question"))
    service = AgenticAnswerService(
        deepseek=model,
        retrieval=_Retrieval(locator),  # type: ignore[arg-type]
        facts=_Facts(_evidence(locator)),
        system_prompt="Read first.",
        max_tool_output_tokens=1,
    )

    async def consume() -> None:
        async for _ in service.stream_answer(
            conversation_id="conversation",
            request=AnswerRequest(question="Question", mode="agentic"),
            history=(),
        ):
            pass

    with pytest.raises(ProviderError) as caught:
        asyncio.run(consume())
    assert caught.value.code == "AGENTIC_TOOL_OUTPUT_LIMIT"
    assert json.loads(model.tool_result)["error"]["code"] == (
        "AGENTIC_TOOL_OUTPUT_LIMIT"
    )


class _RepeatedToolModel:
    def __init__(self, name: str, arguments: dict[str, object]):
        self.name = name
        self.arguments = arguments
        self.calls = 0

    async def tool_turn(self, *, messages, tools):
        self.calls += 1
        raw = {
            "id": f"call-{self.calls}",
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }
        return AssistantToolTurn(
            content=None,
            tool_calls=(
                ToolCall(
                    id=raw["id"],
                    name=self.name,
                    arguments=raw["function"]["arguments"],  # type: ignore[index]
                ),
            ),
            provider_message={
                "role": "assistant",
                "content": None,
                "reasoning_content": f"private-{self.calls}",
                "tool_calls": [raw],
            },
        )


@pytest.mark.parametrize(
    ("model", "service_kwargs", "expected_code"),
    [
        (
            "invalid_window",
            {"max_reads": 12, "max_rounds": 2},
            "AGENTIC_INVALID_WINDOW",
        ),
        (
            "round_limit",
            {"max_rounds": 1},
            "AGENTIC_ROUND_LIMIT",
        ),
    ],
)
def test_agentic_limits_are_typed(
    model: str, service_kwargs: dict[str, int], expected_code: str
) -> None:
    locator = _locator()
    if model == "round_limit":
        tool_model = _RepeatedToolModel("search_keyword", {"query": "x"})
    else:
        window = 9 if model == "invalid_window" else 0
        tool_model = _RepeatedToolModel(
            "read_source_window",
            {"locator": locator.model_dump(mode="json"), "window": window},
        )
    service = AgenticAnswerService(
        deepseek=tool_model,
        retrieval=_Retrieval(locator),  # type: ignore[arg-type]
        facts=_Facts(_evidence(locator, text="x" * 100)),
        system_prompt="Read first.",
        **service_kwargs,
    )

    async def consume() -> None:
        async for _ in service.stream_answer(
            conversation_id="conversation",
            request=AnswerRequest(question="Question", mode="agentic"),
            history=_history(),
        ):
            pass

    with pytest.raises(ProviderError) as caught:
        asyncio.run(consume())
    assert caught.value.code == expected_code


class _BudgetRecoveryModel:
    def __init__(self, locator: DocumentChunkLocator, read_calls: int):
        self.locator = locator
        self.read_calls = read_calls
        self.calls = 0
        self.tool_errors: list[str] = []

    async def tool_turn(self, *, messages, tools):
        self.calls += 1
        if self.calls <= self.read_calls:
            raw = {
                "id": f"read-{self.calls}",
                "type": "function",
                "function": {
                    "name": "read_source_window",
                    "arguments": json.dumps(
                        {"locator": self.locator.model_dump(mode="json"), "window": 0}
                    ),
                },
            }
            return AssistantToolTurn(
                content=None,
                tool_calls=(
                    ToolCall(
                        id=raw["id"],
                        name="read_source_window",
                        arguments=raw["function"]["arguments"],  # type: ignore[index]
                    ),
                ),
                provider_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [raw],
                },
            )
        tool_result = json.loads(messages[-1]["content"])
        self.tool_errors.append(tool_result["error"]["code"])
        answer = (
            "[evidence:source:doc:chunk] 证据不足，无法进一步确认。"
            if self.tool_errors[-1] == "AGENTIC_READ_LIMIT"
            else "证据不足，无法进一步确认。"
        )
        return AssistantToolTurn(
            content=answer,
            tool_calls=(),
            provider_message={"role": "assistant", "content": answer},
        )


@pytest.mark.parametrize(
    ("service_kwargs", "read_calls", "expected_code"),
    [
        ({"max_reads": 1, "max_rounds": 4}, 2, "AGENTIC_READ_LIMIT"),
        ({"max_evidence_tokens": 1, "max_rounds": 3}, 1, "AGENTIC_EVIDENCE_LIMIT"),
    ],
)
def test_agentic_budget_limits_are_recoverable(
    service_kwargs: dict[str, int], read_calls: int, expected_code: str
) -> None:
    locator = _locator()
    model = _BudgetRecoveryModel(locator, read_calls)
    service = AgenticAnswerService(
        deepseek=model,
        retrieval=_Retrieval(locator),  # type: ignore[arg-type]
        facts=_Facts(_evidence(locator, text="x" * 100)),
        system_prompt="Read first.",
        **service_kwargs,
    )

    async def collect():
        return [
            event
            async for event in service.stream_answer(
                conversation_id="conversation",
                request=AnswerRequest(question="问题", mode="agentic"),
                history=(),
            )
        ]

    events = asyncio.run(collect())
    assert model.tool_errors == [expected_code]
    assert events[-1].type == "done"


def test_agentic_tool_registry_is_exact_and_has_no_wiki_or_rescan() -> None:
    names = [tool["function"]["name"] for tool in AGENTIC_TOOLS]
    assert names == [
        "search_keyword",
        "search_semantic",
        "read_source_window",
        "inspect_images",
    ]
    for tool in AGENTIC_TOOLS[:2]:
        assert tool["function"]["parameters"]["required"] == ["query", "limit"]
        assert (
            tool["function"]["parameters"]["properties"]["query"]["maxLength"] == 2_000
        )
    inspect_ids = AGENTIC_TOOLS[3]["function"]["parameters"]["properties"][
        "evidence_ids"
    ]
    assert inspect_ids["maxItems"] == 64
    assert inspect_ids["items"]["maxLength"] == 1024
    assert AGENTIC_MAX_READS == 24


class _BrokenImageObserver:
    async def enrich(self, evidence):
        raise OSError("vision cache failed")


@pytest.mark.parametrize(
    ("observer", "expected_error"),
    [
        (None, "ZHIPU_VISION_NOT_CONFIGURED"),
        (_BrokenImageObserver(), "ZHIPU_VISION_UNAVAILABLE"),
    ],
)
def test_agentic_image_failure_returns_caption_fallback_without_aborting(
    observer, expected_error: str
) -> None:
    image = ImageReference(
        image_id="image-one",
        url="https://cdn.example.com/chart.png",
        caption="Retention chart",
    )
    locator = _locator((image,))
    evidence = _evidence(locator)
    evidence_by_id = OrderedDict(((evidence.evidence_id, evidence),))
    service = AgenticAnswerService(
        deepseek=_FinalModel("unused"),
        retrieval=_Retrieval(locator),  # type: ignore[arg-type]
        facts=_Facts(evidence),
        system_prompt="Read first.",
        image_observer=observer,
    )

    result, _ = asyncio.run(
        service._inspect(
            {"evidence_ids": [evidence.evidence_id]},
            evidence_by_id,
            evidence_tokens=0,
        )
    )

    assert result["warning"]["error"]["code"] == "VISION_UNAVAILABLE"
    observation = result["image_observations"][0]
    assert observation["inspected"] is False
    assert observation["error"] == expected_error
    assert observation["url"] == image.url
    assert observation["caption"] == image.caption
    assert (
        evidence_by_id[evidence.evidence_id].image_observations[0].error
        == expected_error
    )
