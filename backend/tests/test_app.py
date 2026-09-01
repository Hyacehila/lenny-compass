from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ai_pm_contracts import (
    AnswerClaim,
    AnswerResponse,
    DocumentChunkLocator,
    DocumentSegment,
    Evidence,
    SseAnswerDeltaEvent,
    SseDoneEvent,
    SsePhaseEvent,
    SseRetrievalSummaryEvent,
)
from fastapi.testclient import TestClient
from lenny_backend.app import create_app
from lenny_backend.conversations import ConversationStore
from lenny_backend.settings import Settings
from lenny_backend.snapshot import RetrievalSnapshot


def _settings(tmp_path: Path, *, frontend: Path | None = None) -> Settings:
    return Settings(
        _env_file=None,
        repository_root=tmp_path,
        chat_db_path=Path("runtime/chat.sqlite3"),
        facts_root=Path("facts"),
        retrieval_manifest_path=Path("retrieval/manifests/build.json"),
        frontend_dist_path=frontend or Path("missing-dist"),
    )


def _locator() -> DocumentChunkLocator:
    return DocumentChunkLocator(
        document_id="document-one",
        chunk_id="chunk-one",
        ordinal=0,
        segment_ordinal=0,
        fact_document_path="documents/v1/document-one.json",
        raw_path="raw/newsletters/one.md",
        raw_sha256="a" * 64,
        enhanced_path="enhanced/newsletters/one.md",
        enhanced_sha256="c" * 64,
        docling_path="documents/v1/document-one.docling.json",
        docling_sha256="d" * 64,
        chunk_sha256="b" * 64,
        segment=DocumentSegment(
            first_self_ref="#/texts/0",
            last_self_ref="#/texts/0",
            item_count=1,
        ),
        self_refs=("#/texts/0",),
    )


def _app(settings: Settings, **kwargs):
    return create_app(settings, trusted_hosts=("testserver",), **kwargs)


def _sse_payloads(body: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_health_and_conversation_routes(tmp_path: Path) -> None:
    client = TestClient(_app(_settings(tmp_path)))

    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["facts_ready"] is False
    assert health.json()["answer_ready"] is False
    assert health.json()["full_ready"] is False
    assert health.json()["agentic_enabled"] is False

    created = client.post("/api/v1/conversations", json={"title": "Test"})
    assert created.status_code == 201
    conversation_id = created.json()["id"]
    assert client.get("/api/v1/conversations").json()[0]["title"] == "Test"
    assert (
        client.get(f"/api/v1/conversations/{conversation_id}").json()["title"] == "Test"
    )
    renamed = client.patch(
        f"/api/v1/conversations/{conversation_id}", json={"title": "Renamed"}
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed"
    assert (
        client.patch(
            f"/api/v1/conversations/{conversation_id}", json={"title": "  "}
        ).status_code
        == 422
    )
    assert (
        client.patch(
            "/api/v1/conversations/missing", json={"title": "Valid"}
        ).status_code
        == 404
    )
    assert client.get(f"/api/v1/conversations/{conversation_id}/messages").json() == []

    unavailable = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={"question": "What happened?", "mode": "direct"},
    )
    payloads = _sse_payloads(unavailable.text)
    assert len(payloads) == 1
    assert payloads[0]["type"] == "error"
    error = payloads[0]["error"]
    assert error["code"] == "ANSWER_SERVICE_UNAVAILABLE"  # type: ignore[index]
    assert error["retryable"] is False  # type: ignore[index]
    assert error["details"]["run_id"] == unavailable.headers["x-answer-run-id"]  # type: ignore[index]
    assert (
        client.get(f"/api/v1/conversations/{conversation_id}/runs").json()[0]["status"]
        == "failed"
    )
    messages = client.get(f"/api/v1/conversations/{conversation_id}/messages").json()
    assert [(item["role"], item["content"]) for item in messages] == [
        ("user", "What happened?")
    ]
    assert messages[0]["mode"] == "direct"

    agentic = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={"question": "Use tools", "mode": "agentic"},
    )
    assert (
        _sse_payloads(agentic.text)[0]["error"]["code"] == "ANSWER_SERVICE_UNAVAILABLE"
    )  # type: ignore[index]

    deleted = client.delete(f"/api/v1/conversations/{conversation_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {}


def test_degraded_health_does_not_repeat_snapshot_validation(
    tmp_path: Path, monkeypatch
) -> None:
    load_calls: list[tuple[object, object]] = []

    def fail_snapshot_load(cls, manifest_path, *, facts_root):
        load_calls.append((manifest_path, facts_root))
        raise sqlite3.DatabaseError("snapshot validation failed once")

    monkeypatch.setattr(
        RetrievalSnapshot,
        "load",
        classmethod(fail_snapshot_load),
    )
    client = TestClient(_app(_settings(tmp_path)))

    for _ in range(3):
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/index/status").status_code == 503

    assert len(load_calls) == 1


class _AnswerService:
    agentic_available = True

    async def stream_answer(self, *, conversation_id, request, history):
        assert history[-1].content == request.question
        evidence = Evidence(
            evidence_id="source:document-one:chunk-one",
            evidence_kind="source_chunk",
            locator=_locator(),
            text="Grounded source text",
        )
        yield SsePhaseEvent(phase="answer")
        yield SseRetrievalSummaryEvent(
            provider_counts={"keyword": 4, "vector": 3}, fused_count=5
        )
        yield SseAnswerDeltaEvent(
            delta="[evidence:source:document-one:chunk-one] Grounded"
        )
        yield SseAnswerDeltaEvent(delta=" answer")
        yield SseDoneEvent(
            response=AnswerResponse(
                request_id="request-one",
                mode=request.mode,
                answer=("[evidence:source:document-one:chunk-one] Grounded answer"),
                evidence=(evidence,),
                claims=(
                    AnswerClaim(
                        claim_id="claim-1",
                        text="Grounded answer",
                        evidence_ids=(evidence.evidence_id,),
                    ),
                ),
            )
        )


def test_injected_answer_service_persists_only_done_answer(tmp_path: Path) -> None:
    client = TestClient(_app(_settings(tmp_path), answer_service=_AnswerService()))
    conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]

    assert client.get("/api/v1/health").json()["agentic_enabled"] is True

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={
            "question": "Question",
            "mode": "direct",
            "idempotency_key": "request-one",
        },
    )
    payloads = _sse_payloads(response.text)
    assert [item["type"] for item in payloads] == [
        "phase",
        "retrieval_summary",
        "answer_delta",
        "answer_delta",
        "done",
    ]
    assert [item["delta"] for item in payloads if item["type"] == "answer_delta"] == [
        "[evidence:source:document-one:chunk-one] Grounded",
        " answer",
    ]
    messages = client.get(f"/api/v1/conversations/{conversation_id}/messages").json()
    assert [(item["role"], item["content"]) for item in messages] == [
        ("user", "Question"),
        (
            "assistant",
            "[evidence:source:document-one:chunk-one] Grounded answer",
        ),
    ]
    assert [item["mode"] for item in messages] == ["direct", "direct"]
    assert messages[1]["answer_kind"] == "grounded"
    assert messages[1]["claims"][0]["evidence_ids"] == ["source:document-one:chunk-one"]
    assert messages[0]["process_trace"] is None
    assert messages[1]["process_trace"] == {
        "phases": [{"phase": "answer", "message": None}],
        "provider_counts": {"keyword": 4, "vector": 3},
        "fused_count": 5,
        "evidence_count": 1,
    }
    run_id = response.headers["x-answer-run-id"]
    assert payloads[-1]["response"]["request_id"] == run_id
    assert (
        client.get(f"/api/v1/conversations/{conversation_id}/runs").json()[0]["status"]
        == "succeeded"
    )

    replay = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={
            "question": "Question",
            "mode": "direct",
            "idempotency_key": "request-one",
        },
    )
    replay_payloads = _sse_payloads(replay.text)
    assert [item["type"] for item in replay_payloads] == ["done"]
    assert replay_payloads[0]["response"]["request_id"] == run_id
    assert (
        len(client.get(f"/api/v1/conversations/{conversation_id}/messages").json()) == 2
    )


class _FactReader:
    def __init__(self, evidence: Evidence):
        self.evidence = evidence

    def read_source_window(self, locator, window=0):
        assert locator == self.evidence.locator
        assert window == 2
        return (self.evidence,)


class _IncompleteAnswerService:
    agentic_available = False

    async def stream_answer(self, *, conversation_id, request, history):
        yield SsePhaseEvent(phase="retrieve")
        yield SseAnswerDeltaEvent(delta="Partial")


class _BadInsufficientAnswerService:
    agentic_available = False

    async def stream_answer(self, *, conversation_id, request, history):
        yield SseDoneEvent(
            response=AnswerResponse(
                request_id="provider-request",
                mode=request.mode,
                answer_kind="insufficient_evidence",
                answer="The moon is made of cheese.",
            )
        )


class _DoneThenWaitService:
    agentic_available = False

    async def stream_answer(self, *, conversation_id, request, history):
        yield SseDoneEvent(
            response=AnswerResponse(
                request_id="provider-request",
                mode=request.mode,
                answer_kind="insufficient_evidence",
                answer="The available evidence is insufficient to answer this question reliably.",
            )
        )
        await __import__("asyncio").Event().wait()


class _BrokenHistoryStore(ConversationStore):
    def messages(self, conversation_id: str, *, limit: int = 500):
        raise sqlite3.DatabaseError("corrupt history")


class _LeakyInvalidEventService:
    agentic_available = False

    async def stream_answer(self, *, conversation_id, request, history):
        yield {
            "type": "phase",
            "phase": "answer",
            "reasoning_content": "TOP_SECRET_PRIVATE_REASONING",
        }


def test_incomplete_stream_does_not_persist_assistant_message(tmp_path: Path) -> None:
    client = TestClient(
        _app(_settings(tmp_path), answer_service=_IncompleteAnswerService())
    )
    conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={"question": "Question", "mode": "direct"},
    )

    assert [item["type"] for item in _sse_payloads(response.text)] == [
        "phase",
        "error",
    ]
    assert (
        _sse_payloads(response.text)[-1]["error"]["code"] == "ANSWER_STREAM_INCOMPLETE"
    )
    messages = client.get(f"/api/v1/conversations/{conversation_id}/messages").json()
    assert [(item["role"], item["content"]) for item in messages] == [
        ("user", "Question")
    ]
    runs = client.get(f"/api/v1/conversations/{conversation_id}/runs").json()
    assert runs[0]["status"] == "failed"
    assert runs[0]["error_code"] == "ANSWER_STREAM_INCOMPLETE"


def test_insufficient_evidence_kind_cannot_bypass_grounding(tmp_path: Path) -> None:
    client = TestClient(
        _app(_settings(tmp_path), answer_service=_BadInsufficientAnswerService())
    )
    conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={"question": "What happened?", "mode": "direct"},
    )

    payloads = _sse_payloads(response.text)
    assert [item["type"] for item in payloads] == ["error"]
    assert payloads[0]["error"]["code"] == "ANSWER_SERVICE_CONTRACT_ERROR"  # type: ignore[index]
    assert (
        len(client.get(f"/api/v1/conversations/{conversation_id}/messages").json()) == 1
    )


def test_first_done_is_committed_without_waiting_for_provider_eof(
    tmp_path: Path,
) -> None:
    client = TestClient(
        _app(_settings(tmp_path), answer_service=_DoneThenWaitService())
    )
    conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={"question": "What happened?", "mode": "direct"},
    )

    assert [item["type"] for item in _sse_payloads(response.text)] == ["done"]
    assert (
        client.get(f"/api/v1/conversations/{conversation_id}/runs").json()[0]["status"]
        == "succeeded"
    )


def test_history_read_failure_still_terminates_and_fails_run(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = _BrokenHistoryStore(settings.resolved_chat_db_path)
    client = TestClient(
        _app(settings, conversation_store=store, answer_service=_AnswerService())
    )
    conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={"question": "Question", "mode": "direct"},
    )

    payloads = _sse_payloads(response.text)
    assert [item["type"] for item in payloads] == ["error"]
    assert payloads[0]["error"]["code"] == "ANSWER_SERVICE_ERROR"  # type: ignore[index]
    assert store.runs(conversation_id)[0].status == "failed"


def test_contract_validation_errors_never_echo_private_input(tmp_path: Path) -> None:
    client = TestClient(
        _app(_settings(tmp_path), answer_service=_LeakyInvalidEventService())
    )
    conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={"question": "Question", "mode": "direct"},
    )

    assert "TOP_SECRET_PRIVATE_REASONING" not in response.text
    payload = _sse_payloads(response.text)[0]
    assert payload["error"]["code"] == "ANSWER_SERVICE_CONTRACT_ERROR"  # type: ignore[index]


def test_fact_window_endpoint_uses_injected_reader(tmp_path: Path) -> None:
    locator = _locator()
    evidence = Evidence(
        evidence_id="source:document-one:chunk-one",
        evidence_kind="source_chunk",
        locator=locator,
        text="Original fact text",
    )
    client = TestClient(_app(_settings(tmp_path), fact_reader=_FactReader(evidence)))

    response = client.post(
        "/api/v1/facts/read-source-window",
        json={"locator": locator.model_dump(mode="json"), "window": 2},
    )
    assert response.status_code == 200
    assert response.json()["evidence"][0]["text"] == "Original fact text"


def test_frontend_static_and_spa_fallback(tmp_path: Path) -> None:
    dist = tmp_path / "frontend-dist"
    dist.mkdir()
    (dist / "index.html").write_text("<main>app</main>", encoding="utf-8")
    (dist / "version.txt").write_text("one", encoding="utf-8")
    client = TestClient(_app(_settings(tmp_path, frontend=dist)))

    assert client.get("/").text == "<main>app</main>"
    assert client.get("/conversations/one").text == "<main>app</main>"
    assert client.get("/version.txt").text == "one"
    assert client.get("/api/v1/not-a-route").status_code == 404


def test_rejects_untrusted_host_and_accepts_loopback(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url="http://127.0.0.1")

    assert client.get("/api/v1/health").status_code == 200
    assert (
        client.get("/api/v1/health", headers={"Host": "attacker.example"}).status_code
        == 400
    )
