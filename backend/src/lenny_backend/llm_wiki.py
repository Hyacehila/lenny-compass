from __future__ import annotations

import hashlib
import ipaddress
import re
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from ai_pm_contracts import Evidence, LlmWikiPageLocator, RetrievalHit

LLM_WIKI_API_VERSION = "v1"
LLM_WIKI_TESTED_RELEASE = "0.6.9"


class LlmWikiError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class LlmWikiAdapter:
    """Disabled-by-default boundary for nashsu/llm_wiki's local v1 HTTP API.

    This adapter is intentionally not registered in the current answer tools.
    ``rescan_sources`` is a control-plane operation and must remain outside any
    answer-model tool registry.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        base_url: str = "http://127.0.0.1:19828/api/v1",
        token: str | None = None,
        project_id: str | None = None,
        allow_remote: bool = False,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        _validate_base_url(base_url, allow_remote=allow_remote)
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.project_id = project_id
        self.allow_remote = allow_remote
        self.timeout = timeout
        self.transport = transport

    async def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "tested_release": LLM_WIKI_TESTED_RELEASE,
            }
        body = await self._request("GET", "/health", authenticated=False)
        result = _unwrap(body)
        if not isinstance(result, dict):
            raise LlmWikiError("INVALID_LLM_WIKI_RESPONSE", "health is not an object")
        return result

    async def resolve_project(self, identifier: str | None = None) -> dict[str, Any]:
        self._ensure_enabled()
        target = (identifier or self.project_id or "current").strip()
        body = await self._request("GET", "/projects")
        project_payload = _unwrap(body)
        projects = _project_list(project_payload)
        if target == "current":
            current_id = None
            if isinstance(project_payload, dict):
                current_id = project_payload.get(
                    "currentProjectId", project_payload.get("current_project_id")
                )
            if isinstance(current_id, str):
                for project in projects:
                    if project.get("id") == current_id:
                        return project
            for project in projects:
                if any(
                    project.get(key) is True
                    for key in ("current", "isCurrent", "active", "isActive")
                ):
                    return project
            if len(projects) == 1:
                return projects[0]
            raise LlmWikiError(
                "LLM_WIKI_PROJECT_NOT_FOUND", "current LLM Wiki project is ambiguous"
            )
        folded = target.casefold()
        exact: list[dict[str, Any]] = []
        partial: list[dict[str, Any]] = []
        for project in projects:
            values = (project.get("id"), project.get("name"), project.get("path"))
            if any(
                isinstance(value, str) and value.casefold() == folded
                for value in values
            ):
                exact.append(project)
            elif any(
                isinstance(value, str) and folded in value.casefold()
                for value in values
            ):
                partial.append(project)
        if len(exact) == 1:
            return exact[0]
        if len(partial) == 1:
            return partial[0]
        if len(exact) > 1 or len(partial) > 1:
            raise LlmWikiError(
                "LLM_WIKI_PROJECT_AMBIGUOUS",
                f"LLM Wiki project match is ambiguous: {target}",
            )
        raise LlmWikiError(
            "LLM_WIKI_PROJECT_NOT_FOUND", f"LLM Wiki project not found: {target}"
        )

    async def search(
        self,
        query: str,
        *,
        project_id: str | None = None,
        top_k: int = 12,
    ) -> tuple[RetrievalHit, ...]:
        self._ensure_enabled()
        if not query.strip():
            raise LlmWikiError("INVALID_LLM_WIKI_QUERY", "query must be non-empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise LlmWikiError("INVALID_LLM_WIKI_LIMIT", "top_k must be 1..50")
        requested_project = project_id or self.project_id or "current"
        if requested_project == "current":
            resolved_project = await self.resolve_project("current")
            stable_id = resolved_project.get("id")
            if not isinstance(stable_id, str) or not stable_id.strip():
                raise LlmWikiError(
                    "LLM_WIKI_PROJECT_NOT_FOUND",
                    "current project did not resolve to a stable id",
                )
            requested_project = stable_id
        project = _path_segment(requested_project)
        body = await self._request(
            "POST",
            f"/projects/{project}/search",
            json_body={"query": query, "topK": top_k, "includeContent": False},
        )
        result = _unwrap(body)
        response_project_id = _response_project_id(body, result)
        stable_project_id = response_project_id or requested_project
        rows = result.get("results") if isinstance(result, dict) else result
        if not isinstance(rows, list):
            raise LlmWikiError(
                "INVALID_LLM_WIKI_RESPONSE", "search results are not an array"
            )
        hits: list[RetrievalHit] = []
        for row in rows[:top_k]:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                continue
            score = row.get("score", row.get("vectorScore", 0.0))
            try:
                native_score = float(score)
            except (TypeError, ValueError):
                native_score = 0.0
            path = str(row["path"])
            hits.append(
                RetrievalHit(
                    provider="llm_wiki",
                    rank=len(hits) + 1,
                    native_score=native_score,
                    locator=LlmWikiPageLocator(
                        project_id=stable_project_id,
                        path=path,
                        fact_id=str(row["factId"]) if row.get("factId") else None,
                        provider_uri=self._page_uri(
                            stable_project_id, path
                        ),
                    ),
                )
            )
        return tuple(hits)

    async def read_page(self, locator: LlmWikiPageLocator) -> Evidence:
        """Read one generated abstract wiki fact; this has no source-window semantics."""

        self._ensure_enabled()
        project = _path_segment(locator.project_id)
        body = await self._request(
            "GET",
            f"/projects/{project}/files/content",
            params={"path": locator.path},
        )
        result = _unwrap(body)
        if isinstance(result, str):
            content = result
            metadata: dict[str, Any] = {}
        elif isinstance(result, dict) and isinstance(result.get("content"), str):
            content = str(result["content"])
            metadata = {
                key: value
                for key, value in result.items()
                if key in {"mimeType", "updatedAt", "size"}
                and isinstance(value, (str, int, float, bool))
            }
        else:
            raise LlmWikiError(
                "INVALID_LLM_WIKI_RESPONSE", "page content is missing"
            )
        if not content.strip():
            raise LlmWikiError("EMPTY_LLM_WIKI_PAGE", "wiki page is empty")
        revision = hashlib.sha256(content.encode("utf-8")).hexdigest()
        resolved = locator.model_copy(
            update={
                "revision": revision,
                "provider_uri": locator.provider_uri
                or self._page_uri(locator.project_id, locator.path),
            }
        )
        title, headings = _markdown_structure(content, locator.path)
        metadata.update(
            {
                "provider": "llm_wiki",
                "fact_kind": "abstract_fact",
                "window_supported": False,
            }
        )
        identity = hashlib.sha256(
            f"{locator.project_id}\0{locator.path}".encode()
        ).hexdigest()[:16]
        return Evidence(
            evidence_id=f"wiki:{identity}:{revision[:12]}",
            evidence_kind="wiki_fact",
            locator=resolved,
            text=content,
            title=title,
            headings=headings,
            metadata=metadata,
        )

    async def get_graph(self, *, project_id: str | None = None) -> dict[str, Any]:
        self._ensure_enabled()
        project = _path_segment(project_id or self.project_id or "current")
        result = _unwrap(
            await self._request("GET", f"/projects/{project}/graph")
        )
        if not isinstance(result, dict):
            raise LlmWikiError("INVALID_LLM_WIKI_RESPONSE", "graph is not an object")
        return result

    async def rescan_sources(self, *, project_id: str | None = None) -> dict[str, Any]:
        """Control-plane mutation. Never expose this method to an answer model."""

        self._ensure_enabled()
        project = _path_segment(project_id or self.project_id or "current")
        result = _unwrap(
            await self._request("POST", f"/projects/{project}/sources/rescan")
        )
        if not isinstance(result, dict):
            raise LlmWikiError("INVALID_LLM_WIKI_RESPONSE", "rescan result is not an object")
        return result

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            raise LlmWikiError(
                "LLM_WIKI_DISABLED", "LLM Wiki integration is disabled"
            )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        authenticated: bool = True,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        self._ensure_enabled()
        headers: dict[str, str] = {"Accept": "application/json"}
        if authenticated and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self.transport
            ) as client:
                response = await client.request(
                    method,
                    f"{self.base_url}{path}",
                    headers=headers,
                    params=params,
                    json=json_body,
                )
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LlmWikiError(
                "LLM_WIKI_HTTP_ERROR",
                f"LLM Wiki request failed: {exc}",
                retryable=True,
            ) from exc
        if isinstance(body, dict) and body.get("ok") is False:
            raise LlmWikiError(
                str(body.get("code") or "LLM_WIKI_API_ERROR"),
                str(body.get("error") or body.get("message") or "LLM Wiki API error"),
            )
        return body

    def _page_uri(self, project_id: str, path: str) -> str:
        return (
            f"{self.base_url}/projects/{quote(project_id, safe='')}/files/content"
            f"?path={quote(path, safe='')}"
        )


def _validate_base_url(base_url: str, *, allow_remote: bool) -> None:
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("LLM_WIKI_API_BASE_URL must be an HTTP(S) URL")
    if parts.username or parts.password:
        raise ValueError("LLM_WIKI_API_BASE_URL must not contain credentials")
    if not parts.path.rstrip("/").endswith("/api/v1"):
        raise ValueError("LLM_WIKI_API_BASE_URL must end with /api/v1")
    host = parts.hostname.casefold()
    loopback = host == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
    if not loopback and not allow_remote:
        raise ValueError(
            "remote LLM Wiki URLs require LLM_WIKI_ALLOW_REMOTE=true"
        )


def _path_segment(value: str) -> str:
    value = value.strip()
    if not value:
        raise LlmWikiError("INVALID_LLM_WIKI_PROJECT", "project id is empty")
    return quote(value, safe="")


def _unwrap(body: Any) -> Any:
    if not isinstance(body, dict):
        return body
    for key in ("data", "result"):
        if key in body:
            return body[key]
    return body


def _response_project_id(body: Any, result: Any) -> str | None:
    for candidate in (body, result):
        if isinstance(candidate, dict):
            value = candidate.get("projectId", candidate.get("project_id"))
            if isinstance(value, str) and value.strip() and value != "current":
                return value
    return None


def _project_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("projects")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise LlmWikiError(
            "INVALID_LLM_WIKI_RESPONSE", "projects response is not an array"
        )
    return value


_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _markdown_structure(content: str, path: str) -> tuple[str, tuple[str, ...]]:
    headings = tuple(match.group(1) for match in _HEADING.finditer(content))
    title = headings[0] if headings else path.rsplit("/", 1)[-1].removesuffix(".md")
    return title, headings
