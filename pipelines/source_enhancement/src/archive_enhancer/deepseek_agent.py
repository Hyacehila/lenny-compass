from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "chapters": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "sections": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "title": {"type": "string", "minLength": 1},
                                "start_turn": {"type": "integer", "minimum": 0},
                                "end_turn": {"type": "integer", "minimum": 0},
                                "approx_word_count": {"type": "integer", "minimum": 0},
                            },
                            "required": ["title", "start_turn", "end_turn", "approx_word_count"],
                        },
                    },
                },
                "required": ["title", "sections"],
            },
        }
    },
    "required": ["chapters"],
}


SYSTEM_PROMPT = (
    "You are a careful transcript chaptering agent. Read the complete numbered Podcast transcript.\n"
    "Return only one valid JSON object with a chapters array. Do not use Markdown fences. Do not rewrite any transcript text.\n"
    "Create broad English chapter titles and narrower English section titles.\n"
    "Sections should target about 1000 English words, but preserve coherent conversations and never split a speaker turn.\n"
    "Every turn must be covered exactly once, in order, from turn 0 through the final turn.\n"
    "Use only turn IDs from the input. Do not invent timestamps, speakers, facts, or quotations.\n"
    "Use these exact property names: chapters, title, sections, start_turn, end_turn, approx_word_count.\n"
    "Do not use alternatives such as start_turn_id, end_turn_id, start, end, word_count, or summary."
)


class AgentError(RuntimeError):
    pass


class NonRetryableAgentError(AgentError):
    pass


class DeepSeekAgent:
    def __init__(self, model: str = "deepseek-v4-flash", max_retries: int = 3, timeout: float = 240.0):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise AgentError("DEEPSEEK_API_KEY is not set")

    def plan(
        self,
        *,
        file_name: str,
        metadata: dict[str, Any],
        transcript: str,
        review: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request: dict[str, Any] = {
            "file": file_name,
            "metadata": {key: metadata.get(key) for key in ("title", "guest", "date", "tags") if metadata.get(key) is not None},
            "required_output_schema": PLAN_SCHEMA,
            "transcript": transcript,
        }
        if review is not None:
            request["previous_plan"] = review["plan"]
            request["validation_warnings"] = review["warnings"]
            request["instruction"] = "Repair the previous chapter plan and return a complete replacement plan."
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        "https://api.deepseek.com/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json=payload,
                    )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise AgentError(f"retryable HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise NonRetryableAgentError(f"HTTP {response.status_code} {response.reason_phrase}")
                response.raise_for_status()
                data = response.json()
                output_text = self._extract_text(data)
                try:
                    plan = self._parse_json_object(output_text)
                except json.JSONDecodeError as exc:
                    raise AgentError(f"DeepSeek output was not JSON (length={len(output_text)})") from exc
                return self._normalize_plan(plan), self._usage(data)
            except NonRetryableAgentError as exc:
                raise AgentError(f"DeepSeek request failed for {file_name}: {exc}") from exc
            except (httpx.HTTPError, json.JSONDecodeError, AgentError, KeyError, TypeError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2**attempt)
        raise AgentError(f"DeepSeek request failed for {file_name}: {last_error}")

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise AgentError("DeepSeek Chat Completions response had no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        raise AgentError("DeepSeek Chat Completions response had empty message content")

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
            if text.endswith("```"):
                text = text[:-3].rstrip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise AgentError("DeepSeek output root was not a JSON object")
        return value

    @staticmethod
    def _normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
        chapters = plan.get("chapters")
        if not isinstance(chapters, list):
            return plan
        normalized: list[Any] = []
        for chapter in chapters:
            if not isinstance(chapter, dict):
                normalized.append(chapter)
                continue
            sections = chapter.get("sections")
            normalized_sections: list[Any] = []
            if isinstance(sections, list):
                for section in sections:
                    if not isinstance(section, dict):
                        normalized_sections.append(section)
                        continue
                    normalized_sections.append(
                        {
                            "title": section.get("title", section.get("section_title")),
                            "start_turn": section.get("start_turn", section.get("start_turn_id", section.get("start"))),
                            "end_turn": section.get("end_turn", section.get("end_turn_id", section.get("end"))),
                            "approx_word_count": section.get(
                                "approx_word_count", section.get("word_count", section.get("estimated_word_count", 0))
                            ),
                        }
                    )
            normalized.append(
                {
                    "title": chapter.get("title", chapter.get("chapter_title")),
                    "sections": normalized_sections if isinstance(sections, list) else sections,
                }
            )
        return {"chapters": normalized}

    @staticmethod
    def _usage(data: dict[str, Any]) -> dict[str, Any]:
        usage = data.get("usage") or {}
        return {
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
            "total_tokens": usage.get("total_tokens"),
        }
