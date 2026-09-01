from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from ai_pm_contracts import (
    AnswerRequest,
    AnswerResponse,
    DocumentChunkLocator,
    DocumentSegment,
    Evidence,
    RetrievalHit,
    SseAnswerDeltaEvent,
    SseDoneEvent,
)
from fastapi.testclient import TestClient
from lenny_backend.agentic import AgenticAnswerService
from lenny_backend.app import create_app
from lenny_backend.conversations import ConversationStore, Message
from lenny_backend.direct import DirectAnswerService, SearchOutcome, _diversify_hits
from lenny_backend.providers import (
    AssistantToolTurn,
    DeepSeekClient,
    ProviderError,
    ToolCall,
)
from lenny_backend.settings import Settings
from lenny_backend.stage2 import (
    AnswerDraft,
    ClaimVerdict,
    DraftAdvice,
    DraftClaim,
    QuestionPlan,
    VerificationBatch,
    coaching_state,
    finalize_stage2_answer,
)


def _locator(document_id: str = "doc", ordinal: int = 0) -> DocumentChunkLocator:
    return DocumentChunkLocator(
        document_id=document_id,
        chunk_id=f"chunk-{ordinal}",
        ordinal=ordinal,
        segment_ordinal=ordinal,
        fact_document_path=f"documents/v1/{document_id}.json",
        raw_path=f"raw/{document_id}.md",
        raw_sha256="a" * 64,
        enhanced_path=f"enhanced/{document_id}.md",
        enhanced_sha256="b" * 64,
        docling_path=f"documents/v1/{document_id}.docling.json",
        docling_sha256="c" * 64,
        chunk_sha256=(f"{ordinal:x}" * 64)[:64],
        segment=DocumentSegment(
            first_self_ref=f"#/texts/{ordinal}",
            last_self_ref=f"#/texts/{ordinal}",
            item_count=1,
        ),
        self_refs=(f"#/texts/{ordinal}",),
    )


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="source:doc:chunk-0",
        evidence_kind="source_chunk",
        locator=_locator(),
        text="Teams should select a metric that reflects durable customer value.",
        title="Metric selection",
    )


def _history() -> tuple[Message, ...]:
    return (
        Message(
            id="user-message",
            conversation_id="conversation",
            role="user",
            content="该怎么选指标？",
            created_at="2026-01-01T00:00:00Z",
        ),
    )


class _RepairingProvider:
    def __init__(self, *, fail_core: bool = False, verifier_error: bool = False):
        self.fail_core = fail_core
        self.verifier_error = verifier_error
        self.verify_calls = 0
        self.repair_calls = 0

    async def draft_stage2_answer(self, **_: Any) -> AnswerDraft:
        return self.initial_draft()

    def initial_draft(self) -> AnswerDraft:
        return AnswerDraft(
            response_kind="grounded_answer",
            claims=(
                DraftClaim(
                    claim_id="core",
                    text="资料建议选择能反映长期客户价值的指标。",
                    role="core",
                    evidence_ids=("source:doc:chunk-0",),
                ),
                DraftClaim(
                    claim_id="weak",
                    text="所有团队都必须采用同一个指标。",
                    role="supporting",
                    evidence_ids=("source:doc:chunk-0",),
                ),
                DraftClaim(
                    claim_id="rationale",
                    text="指标应反映长期客户价值。",
                    role="advice_rationale",
                    evidence_ids=("source:doc:chunk-0",),
                ),
            ),
            advice=(
                DraftAdvice(
                    advice_id="local-advice",
                    text="先写清楚用户价值，再筛指标。",
                    basis="evidence_synthesis",
                    rationale_claim_ids=("rationale",),
                ),
                DraftAdvice(
                    advice_id="general-advice",
                    text="可以先开一个小范围评审会。",
                    basis="general_pm_guidance",
                ),
            ),
        )

    async def verify_stage2_answer(
        self, *, draft: AnswerDraft, **_: Any
    ) -> VerificationBatch:
        self.verify_calls += 1
        if self.verifier_error:
            raise ProviderError(
                "SEMANTIC_VERIFICATION_UNAVAILABLE",
                "verifier unavailable",
                retryable=True,
            )
        verdicts = []
        for claim in draft.claims:
            failed = claim.claim_id == "weak" or (
                self.fail_core and claim.claim_id == "core"
            )
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim.claim_id,
                    verdict="insufficient_context" if failed else "supported",
                )
            )
        return VerificationBatch(verdicts=tuple(verdicts))

    async def repair_stage2_answer(
        self, *, draft: AnswerDraft, verification: VerificationBatch, **_: Any
    ) -> AnswerDraft:
        self.repair_calls += 1
        failed = {
            item.claim_id
            for item in verification.verdicts
            if item.verdict != "supported"
        }
        if "core" in failed:
            general = tuple(
                item for item in draft.advice if item.basis == "general_pm_guidance"
            )
            return AnswerDraft(response_kind="general_guidance", advice=general)
        return draft.model_copy(
            update={"claims": tuple(item for item in draft.claims if item.claim_id not in failed)}
        )


def test_stage2_repairs_once_and_publishes_only_supported_claims() -> None:
    provider = _RepairingProvider()

    response = asyncio.run(
        finalize_stage2_answer(
            provider=provider,
            mode="direct",
            request_id="request",
            question="该怎么选指标？",
            conversation_context="",
            evidence=(_evidence(),),
            history=_history(),
            initial_draft=provider.initial_draft(),
        )
    )

    assert response.contract_version == "v2"
    assert response.response_kind == "grounded_answer"
    assert {item.claim_id for item in response.claims} == {"core", "rationale"}
    assert response.verification is not None
    assert response.verification.checked_claim_count == 2
    assert response.verification.omitted_claim_count == 1
    assert response.verification.calibration_level == "uncalibrated"
    assert provider.verify_calls == 2
    assert provider.repair_calls == 1


def test_stage2_core_failure_downgrades_to_clearly_labeled_general_guidance() -> None:
    provider = _RepairingProvider(fail_core=True)

    response = asyncio.run(
        finalize_stage2_answer(
            provider=provider,
            mode="direct",
            request_id="request",
            question="该怎么选指标？",
            conversation_context="",
            evidence=(_evidence(),),
            history=_history(),
            initial_draft=provider.initial_draft(),
        )
    )

    assert response.response_kind == "general_guidance"
    assert response.answer_kind is None
    assert response.evidence == ()
    assert response.claims == ()
    assert [item.advice_id for item in response.advice] == ["general-advice"]
    assert "不代表资料库结论" in response.answer


def test_stage2_verifier_outage_refuses_instead_of_publishing_unchecked_claims() -> None:
    provider = _RepairingProvider(verifier_error=True)

    response = asyncio.run(
        finalize_stage2_answer(
            provider=provider,
            mode="agentic",
            request_id="request",
            question="该怎么选指标？",
            conversation_context="",
            evidence=(_evidence(),),
            history=_history(),
            initial_draft=provider.initial_draft(),
        )
    )

    assert response.response_kind == "insufficient_evidence"
    assert response.insufficient_reason == "semantic_support_uncertain"
    assert response.evidence == ()
    assert response.claims == ()


def test_coaching_reply_must_reference_latest_persisted_clarification() -> None:
    assistant = Message(
        id="assistant-message",
        conversation_id="conversation",
        role="assistant",
        content="你当前更在意增长还是留存？",
        created_at="2026-01-01T00:00:01Z",
        contract_version="v2",
        response_kind="clarification",
        response_payload={
            "clarification": {
                "coaching_id": "coach-one",
                "round": 2,
                "prompt": "你当前更在意增长还是留存？",
                "suggestions": ["增长", "留存"],
                "can_skip": True,
            }
        },
    )
    current_user = _history()[0]

    assert coaching_state(
        (assistant, current_user),
        coaching_id="coach-one",
        reply_to_message_id="assistant-message",
    ) == (2, "coach-one")
    try:
        coaching_state(
            (assistant, current_user),
            coaching_id="coach-one",
            reply_to_message_id="stale-message",
        )
    except ProviderError as exc:
        assert exc.code == "INVALID_COACHING_CONTEXT"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("stale coaching context was accepted")


def test_broad_retrieval_diversifies_documents_without_reordering_hits() -> None:
    hits = tuple(
        RetrievalHit(
            provider="keyword",
            rank=index + 1,
            native_score=float(100 - index),
            locator=_locator("dominant" if index < 5 else f"doc-{index}", index),
        )
        for index in range(8)
    )

    selected = _diversify_hits(hits, limit=6, per_document=3)

    assert [item.locator.chunk_id for item in selected] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
        "chunk-5",
        "chunk-6",
        "chunk-7",
    ]


class _Stage2DirectProvider:
    def __init__(self, *, clarify: bool = False):
        self.clarify = clarify

    async def plan_question(self, **_: Any) -> QuestionPlan:
        return QuestionPlan(
            intent="scenario_advice",
            standalone_question="如何选择指标？",
            semantic_query_en="how to choose a product metric",
            lexical_terms_en=("metric", "customer value"),
            source_scope="broad",
            needs_clarification=self.clarify,
            clarification_prompt="你更想改善激活还是留存？" if self.clarify else None,
            clarification_suggestions=("激活", "留存") if self.clarify else (),
        )

    async def draft_stage2_answer(self, **_: Any) -> AnswerDraft:
        return AnswerDraft(
            response_kind="grounded_answer",
            claims=(
                DraftClaim(
                    claim_id="core",
                    text="资料建议选择能反映长期客户价值的指标。",
                    role="core",
                    evidence_ids=("source:doc:chunk-0",),
                ),
            ),
        )

    async def verify_stage2_answer(
        self, *, draft: AnswerDraft, **_: Any
    ) -> VerificationBatch:
        return VerificationBatch(
            verdicts=tuple(
                ClaimVerdict(claim_id=item.claim_id, verdict="supported")
                for item in draft.claims
            )
        )

    async def repair_stage2_answer(self, *, draft: AnswerDraft, **_: Any) -> AnswerDraft:
        return draft


class _Retrieval:
    async def search(self, _: Any) -> SearchOutcome:
        return SearchOutcome(
            keyword_hits=(
                RetrievalHit(
                    provider="keyword",
                    rank=1,
                    native_score=1.0,
                    locator=_evidence().locator,
                ),
            ),
            vector_hits=(),
            errors=("SEMANTIC_PROVIDER_UNAVAILABLE",),
        )


class _Facts:
    def read_source_window(self, _: Any, window: int = 0) -> tuple[Evidence, ...]:
        assert window in {0, 1}
        return (_evidence(),)


def test_direct_stage2_runs_plan_draft_verifier_and_returns_v2_contract() -> None:
    service = DirectAnswerService(
        deepseek=_Stage2DirectProvider(),  # type: ignore[arg-type]
        retrieval=_Retrieval(),
        facts=_Facts(),
    )

    async def collect() -> list[Any]:
        return [
            item
            async for item in service.stream_answer(
                conversation_id="conversation",
                request=AnswerRequest(question="如何选择指标？", mode="direct"),
                history=_history(),
            )
        ]

    events = asyncio.run(collect())
    done = next(item for item in events if isinstance(item, SseDoneEvent))
    assert done.response.contract_version == "v2"
    assert done.response.response_kind == "grounded_answer"
    assert [item.phase for item in events if hasattr(item, "phase")] == [
        "plan",
        "retrieve",
        "resolve",
        "answer",
        "verify",
    ]


def test_direct_stage2_clarification_short_circuits_retrieval() -> None:
    service = DirectAnswerService(
        deepseek=_Stage2DirectProvider(clarify=True),  # type: ignore[arg-type]
        retrieval=_Retrieval(),
        facts=_Facts(),
    )

    async def collect() -> list[Any]:
        return [
            item
            async for item in service.stream_answer(
                conversation_id="conversation",
                request=AnswerRequest(question="帮我选一个指标", mode="direct"),
                history=_history(),
            )
        ]

    events = asyncio.run(collect())
    assert isinstance(events[-2], SseAnswerDeltaEvent)
    assert isinstance(events[-1], SseDoneEvent)
    assert events[-1].response.response_kind == "clarification"
    assert not any(getattr(item, "phase", None) == "retrieve" for item in events)


class _AgenticStage2Provider(_Stage2DirectProvider):
    def __init__(self):
        super().__init__()
        self.turn = 0

    async def tool_turn(self, **_: Any) -> AssistantToolTurn:
        self.turn += 1
        if self.turn == 1:
            call = ToolCall(
                id="read-one",
                name="read_source_window",
                arguments=json.dumps(
                    {"locator": _evidence().locator.model_dump(mode="json"), "window": 0}
                ),
            )
            return AssistantToolTurn(
                content=None,
                tool_calls=(call,),
                provider_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                    ],
                },
            )
        return AssistantToolTurn(
            content='{"research_complete":true}',
            tool_calls=(),
            provider_message={
                "role": "assistant",
                "content": '{"research_complete":true}',
            },
        )


class _AgenticRetrieval:
    async def keyword(self, _: str, *, limit: int = 50) -> tuple[RetrievalHit, ...]:
        assert limit == 6
        return (
            RetrievalHit(
                provider="keyword",
                rank=1,
                native_score=1.0,
                locator=_evidence().locator,
            ),
        )

    async def semantic(self, _: str, *, limit: int = 50) -> tuple[RetrievalHit, ...]:
        assert limit == 6
        return ()


def test_agentic_stage2_research_handoff_is_written_and_verified_separately() -> None:
    provider = _AgenticStage2Provider()
    service = AgenticAnswerService(
        deepseek=provider,  # type: ignore[arg-type]
        retrieval=_AgenticRetrieval(),  # type: ignore[arg-type]
        facts=_Facts(),
        system_prompt="Research local evidence.",
    )

    async def collect() -> list[Any]:
        return [
            item
            async for item in service.stream_answer(
                conversation_id="conversation",
                request=AnswerRequest(question="如何选择指标？", mode="agentic"),
                history=_history(),
            )
        ]

    events = asyncio.run(collect())
    done = next(item for item in events if isinstance(item, SseDoneEvent))
    assert provider.turn == 2
    assert done.response.contract_version == "v2"
    assert done.response.response_kind == "grounded_answer"
    assert done.response.evidence[0].evidence_id == "source:doc:chunk-0"


class _ClarificationService:
    async def stream_answer(self, **_: Any):
        response = AnswerResponse(
            request_id="service-request",
            mode="direct",
            contract_version="v2",
            response_kind="clarification",
            answer_kind=None,
            answer="你更想改善激活还是留存？",
            clarification={
                "coaching_id": "coach-one",
                "round": 1,
                "prompt": "你更想改善激活还是留存？",
                "suggestions": ["激活", "留存"],
            },
        )
        yield SseAnswerDeltaEvent(delta=response.answer)
        yield SseDoneEvent(response=response)


def test_app_persists_and_replays_stage2_clarification(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        repository_root=tmp_path,
        chat_db_path=Path("runtime/chat.sqlite3"),
        facts_root=Path("facts"),
        retrieval_manifest_path=Path("retrieval/manifests/build.json"),
        frontend_dist_path=Path("missing"),
    )
    store = ConversationStore(settings.resolved_chat_db_path)
    app = create_app(
        settings,
        conversation_store=store,
        answer_service=_ClarificationService(),
        trusted_hosts=("testserver",),
    )
    client = TestClient(app)
    conversation_id = client.post(
        "/api/v1/conversations", json={"title": "Stage 2"}
    ).json()["id"]
    request = {
        "question": "帮我选一个指标",
        "mode": "direct",
        "idempotency_key": "stage2-replay",
    }

    first = client.post(f"/api/v1/conversations/{conversation_id}/ask", json=request)
    second = client.post(f"/api/v1/conversations/{conversation_id}/ask", json=request)
    first_done = [
        line for line in first.text.splitlines() if '"type":"done"' in line
    ][0]
    second_done = [
        line for line in second.text.splitlines() if '"type":"done"' in line
    ][0]

    assert '"contract_version":"v2"' in first_done
    assert '"response_kind":"clarification"' in second_done
    messages = client.get(
        f"/api/v1/conversations/{conversation_id}/messages"
    ).json()
    assistant = messages[-1]
    assert assistant["contract_version"] == "v2"
    assert assistant["response_payload"]["clarification"]["coaching_id"] == "coach-one"
    assert assistant["id"] in first_done
    assert assistant["id"] in second_done


def test_deepseek_stage2_uses_a_distinct_verifier_call_and_strict_json() -> None:
    payloads: list[dict[str, Any]] = []
    responses = iter(
        (
            {
                "intent": "knowledge",
                "standalone_question": "How should a metric be selected?",
                "semantic_query_en": "product metric selection",
                "lexical_terms_en": ["metric"],
                "lexical_phrases_en": [],
                "source_scope": "broad",
                "needs_clarification": False,
                "clarification_prompt": None,
                "clarification_suggestions": [],
            },
            {
                "response_kind": "grounded_answer",
                "claims": [
                    {
                        "claim_id": "core",
                        "text": "Choose a metric that reflects durable customer value.",
                        "role": "core",
                        "evidence_ids": ["source:doc:chunk-0"],
                    }
                ],
                "advice": [],
                "user_premises": [],
            },
            {
                "verdicts": [{"claim_id": "core", "verdict": "supported"}],
                "general_guidance_safe": True,
            },
            {
                "response_kind": "grounded_answer",
                "claims": [
                    {
                        "claim_id": "core",
                        "text": "Choose a metric that reflects durable customer value.",
                        "role": "core",
                        "evidence_ids": ["source:doc:chunk-0"],
                    }
                ],
                "advice": [],
                "user_premises": [],
            },
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(next(responses), ensure_ascii=False)
                        },
                    }
                ]
            },
        )

    client = DeepSeekClient(
        api_key="test",
        base_url="https://deepseek.test/v1",
        rewrite_model="planner-model",
        answer_model="writer-model",
        verify_model="verifier-model",
        rewrite_prompt="legacy rewrite",
        answer_prompt="legacy answer",
        plan_prompt="question planner",
        stage2_answer_prompt="answer writer",
        verify_prompt="independent verifier",
        repair_prompt="single repair",
        transport=httpx.MockTransport(handler),
    )

    async def run_calls() -> None:
        plan = await client.plan_question(
            question="How?",
            conversation_context="",
            completed_clarification_rounds=0,
            skip_clarification=False,
        )
        draft = await client.draft_stage2_answer(
            question="How?",
            conversation_context="",
            evidence=(_evidence(),),
            message_ids=("user-message",),
        )
        verification = await client.verify_stage2_answer(
            draft=draft,
            evidence=(_evidence(),),
            conversation_context="",
        )
        repaired = await client.repair_stage2_answer(
            question="How?",
            draft=draft,
            verification=verification,
            evidence=(_evidence(),),
            conversation_context="",
            message_ids=("user-message",),
        )
        assert plan.intent == "knowledge"
        assert repaired.claims[0].claim_id == "core"

    asyncio.run(run_calls())

    assert [payload["model"] for payload in payloads] == [
        "planner-model",
        "writer-model",
        "verifier-model",
        "writer-model",
    ]
    assert [payload["messages"][0]["content"] for payload in payloads] == [
        "question planner",
        "answer writer",
        "independent verifier",
        "single repair",
    ]
    assert all(payload["response_format"] == {"type": "json_object"} for payload in payloads)
