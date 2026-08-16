from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from ai_pm_contracts import LlmWikiPageLocator
from lenny_backend.llm_wiki import LlmWikiAdapter, LlmWikiError


def test_disabled_adapter_never_performs_network_io() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("disabled adapter must not make requests")

    adapter = LlmWikiAdapter(
        enabled=False,
        transport=httpx.MockTransport(handler),
    )
    health = asyncio.run(adapter.health())
    assert health["status"] == "disabled"
    with pytest.raises(LlmWikiError) as caught:
        asyncio.run(adapter.search("query"))
    assert caught.value.code == "LLM_WIKI_DISABLED"
    assert calls == 0


def test_remote_url_guard_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="ALLOW_REMOTE"):
        LlmWikiAdapter(enabled=False, base_url="https://wiki.example.com/api/v1")
    adapter = LlmWikiAdapter(
        enabled=False,
        base_url="https://wiki.example.com/api/v1",
        allow_remote=True,
    )
    assert asyncio.run(adapter.health())["status"] == "disabled"
    with pytest.raises(ValueError, match="/api/v1"):
        LlmWikiAdapter(enabled=False, base_url="http://127.0.0.1:19828/wrong")


def test_official_v1_paths_auth_shapes_and_abstract_fact_read() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/api/v1/health":
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"ok": True, "status": "running"})
        assert request.headers["authorization"] == "Bearer local-token"
        if path == "/api/v1/projects":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "data": {
                        "currentProjectId": "p1",
                        "projects": [
                            {"id": "p1", "name": "Product Management Wiki"},
                            {"id": "p2", "name": "Other"},
                        ],
                    },
                },
            )
        if path == "/api/v1/projects/p1/search":
            assert json.loads(request.content) == {
                "query": "retention",
                "topK": 3,
                "includeContent": False,
            }
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "data": {
                        "mode": "hybrid",
                        "tokenHits": 1,
                        "vectorHits": 1,
                        "results": [
                            {
                                "path": "wiki/concepts/cohort retention.md",
                                "title": "Cohort retention",
                                "score": 8.5,
                            }
                        ],
                    },
                },
            )
        if path == "/api/v1/projects/p1/files/content":
            assert request.url.params["path"] == "wiki/concepts/cohort retention.md"
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "data": {
                        "path": "wiki/concepts/cohort retention.md",
                        "content": "# Cohort retention\n\nCompiled abstract fact.",
                        "updatedAt": "2026-01-01T00:00:00Z",
                    },
                },
            )
        if path == "/api/v1/projects/p1/graph":
            return httpx.Response(200, json={"ok": True, "data": {"nodes": [], "edges": []}})
        if path == "/api/v1/projects/p1/sources/rescan":
            assert request.content == b""
            return httpx.Response(200, json={"ok": True, "data": {"queued": 2}})
        raise AssertionError(f"unexpected path: {path}")

    adapter = LlmWikiAdapter(
        enabled=True,
        token="local-token",
        project_id="p1",
        transport=httpx.MockTransport(handler),
    )

    async def exercise():
        health = await adapter.health()
        current = await adapter.resolve_project("current")
        substring = await adapter.resolve_project("management")
        hits = await adapter.search("retention", top_k=3)
        fact = await adapter.read_page(hits[0].locator)
        graph = await adapter.get_graph()
        rescan = await adapter.rescan_sources()
        return health, current, substring, hits, fact, graph, rescan

    health, current, substring, hits, fact, graph, rescan = asyncio.run(exercise())
    assert health["status"] == "running"
    assert current["id"] == "p1"
    assert substring["id"] == "p1"
    assert hits[0].locator == LlmWikiPageLocator(
        project_id="p1",
        path="wiki/concepts/cohort retention.md",
        provider_uri=(
            "http://127.0.0.1:19828/api/v1/projects/p1/files/content"
            "?path=wiki%2Fconcepts%2Fcohort%20retention.md"
        ),
    )
    assert fact.evidence_kind == "wiki_fact"
    assert fact.metadata["fact_kind"] == "abstract_fact"
    assert fact.metadata["window_supported"] is False
    assert fact.locator.project_id == "p1"
    assert fact.locator.path == "wiki/concepts/cohort retention.md"
    assert fact.locator.revision is not None
    assert graph == {"nodes": [], "edges": []}
    assert rescan == {"queued": 2}
    assert not hasattr(adapter, "read_source_window")


def test_search_resolves_current_to_stable_project_and_enforces_topk_50() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/projects":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "currentProjectId": "resolved-project",
                        "projects": [
                            {"id": "resolved-project", "name": "Current"}
                        ],
                    }
                },
            )
        assert request.url.path == "/api/v1/projects/resolved-project/search"
        return httpx.Response(
            200,
            json={
                "data": {
                    "projectId": "response-project-id",
                    "results": [{"path": "wiki/page.md", "score": 1}],
                }
            },
        )

    adapter = LlmWikiAdapter(
        enabled=True,
        transport=httpx.MockTransport(handler),
    )
    hits = asyncio.run(adapter.search("query", top_k=50))
    assert hits[0].locator.project_id == "response-project-id"
    assert "/projects/response-project-id/" in hits[0].locator.provider_uri
    with pytest.raises(LlmWikiError) as caught:
        asyncio.run(adapter.search("query", top_k=51))
    assert caught.value.code == "INVALID_LLM_WIKI_LIMIT"
