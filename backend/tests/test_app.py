from __future__ import annotations

import json
from pathlib import Path

from ai_pm_contracts import (
    AnswerResponse,
    DocumentChunkLocator,
    DocumentSegment,
    Evidence,
    SseAnswerDeltaEvent,
    SseDoneEvent,
    SsePhaseEvent,
)
from fastapi.testclient import TestClient
from lenny_backend.app import create_app
from lenny_backend.settings import Settings


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
    assert client.get(f"/api/v1/conversations/{conversation_id}").json()["title"] == "Test"
    renamed = client.patch(
        f"/api/v1/conversations/{conversation_id}", json={"title": "Renamed"}
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed"
    assert client.patch(
        f"/api/v1/conversations/{conversation_id}", json={"title": "  "}
    ).status_code == 422
    assert client.patch(
        "/api/v1/conversations/missing", json={"title": "Valid"}
    ).status_code == 404
    assert client.get(f"/api/v1/conversations/{conversation_id}/messages").json() == []

    unavailable = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={"question": "What happened?", "mode": "direct"},
    )
    payloads = _sse_payloads(unavailable.text)
    assert payloads == [
        {
            "type": "error",
            "error": {
                "code": "ANSWER_SERVICE_UNAVAILABLE",
                "message": "No answer service is configured",
                "retryable": False,
                "details": {},
            },
        }
    ]
    messages = client.get(
        f"/api/v1/conversations/{conversation_id}/messages"
    ).json()
    assert [(item["role"], item["content"]) for item in messages] == [
        ("user", "What happened?")
    ]
    assert messages[0]["mode"] == "direct"

    agentic = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={"question": "Use tools", "mode": "agentic"},
    )
    assert _sse_payloads(agentic.text)[0]["error"]["code"] == "ANSWER_SERVICE_UNAVAILABLE"  # type: ignore[index]

    deleted = client.delete(f"/api/v1/conversations/{conversation_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {}


class _AnswerService:
    agentic_available = True

    async def stream_answer(self, *, conversation_id, request, history):
        assert history[-1].content == request.question
        yield SsePhaseEvent(phase="answer")
        yield SseAnswerDeltaEvent(delta="Grounded")
        yield SseAnswerDeltaEvent(delta=" answer")
        yield SseDoneEvent(
            response=AnswerResponse(
                request_id="request-one",
                mode=request.mode,
                answer="Grounded answer",
            )
        )


def test_injected_answer_service_persists_only_done_answer(tmp_path: Path) -> None:
    client = TestClient(
        _app(_settings(tmp_path), answer_service=_AnswerService())
    )
    conversation_id = client.post("/api/v1/conversations", json={}).json()["id"]

    assert client.get("/api/v1/health").json()["agentic_enabled"] is True

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={"question": "Question", "mode": "direct"},
    )
    payloads = _sse_payloads(response.text)
    assert [item["type"] for item in payloads] == [
        "phase",
        "answer_delta",
        "answer_delta",
        "done",
    ]
    assert [item["delta"] for item in payloads if item["type"] == "answer_delta"] == [
        "Grounded",
        " answer",
    ]
    messages = client.get(
        f"/api/v1/conversations/{conversation_id}/messages"
    ).json()
    assert [(item["role"], item["content"]) for item in messages] == [
        ("user", "Question"),
        ("assistant", "Grounded answer"),
    ]
    assert [item["mode"] for item in messages] == ["direct", "direct"]


class _FactReader:
    def __init__(self, evidence: Evidence):
        self.evidence = evidence

    def read_source_window(self, locator, window=0):
        assert locator == self.evidence.locator
        assert window == 2
        return (self.evidence,)


def test_fact_window_endpoint_uses_injected_reader(tmp_path: Path) -> None:
    locator = _locator()
    evidence = Evidence(
        evidence_id="source:document-one:chunk-one",
        evidence_kind="source_chunk",
        locator=locator,
        text="Original fact text",
    )
    client = TestClient(
        _app(_settings(tmp_path), fact_reader=_FactReader(evidence))
    )

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
    assert client.get(
        "/api/v1/health", headers={"Host": "attacker.example"}
    ).status_code == 400
