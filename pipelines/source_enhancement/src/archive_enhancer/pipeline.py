from __future__ import annotations

import json
import hashlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from .frontmatter import document_heading, split_front_matter
from .renderer import render_podcast
from .transcript import numbered_transcript, parse_turns, word_count
from .validator import validate_document


_WRITE_LOCK = Lock()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _plan_warnings(plan: dict[str, Any], turns: list[Any]) -> list[str]:
    warnings: list[str] = []
    chapters = plan.get("chapters") if isinstance(plan, dict) else None
    if not isinstance(chapters, list) or not chapters:
        return ["HARD:no chapters"]
    sections: list[Any] = []
    seen_chapter_titles: set[str] = set()
    for chapter in chapters:
        if not isinstance(chapter, dict):
            warnings.append("HARD:chapter is not an object")
            continue
        chapter_title = str(chapter.get("title") or "").strip()
        if not chapter_title or chapter_title.startswith("#"):
            warnings.append("HARD:empty chapter title")
        elif chapter_title.casefold() in seen_chapter_titles:
            warnings.append("HARD:duplicate chapter title")
        seen_chapter_titles.add(chapter_title.casefold())
        chapter_sections = chapter.get("sections")
        if not isinstance(chapter_sections, list) or not chapter_sections:
            warnings.append("HARD:chapter has no sections")
            continue
        for section in chapter_sections:
            sections.append(section)
    if not sections:
        return warnings + ["HARD:no sections"]
    expected = 0
    seen_titles: set[str] = set()
    for section in sections:
        if not isinstance(section, dict):
            warnings.append("HARD:section is not an object")
            continue
        title = str(section.get("title") or "").strip()
        if not title or title.startswith("#"):
            warnings.append("HARD:invalid section title")
        if title.casefold() in seen_titles:
            warnings.append("HARD:duplicate section title")
        seen_titles.add(title.casefold())
        try:
            start = int(section.get("start_turn", -1))
            end = int(section.get("end_turn", -1))
        except (TypeError, ValueError):
            warnings.append("HARD:turn range is not numeric")
            continue
        if start != expected:
            warnings.append(f"HARD:coverage gap or overlap at turn {expected}; found {start}")
        if end < start or end >= len(turns):
            warnings.append(f"HARD:invalid turn range {start}-{end}")
        if 0 <= start <= end < len(turns):
            size = sum(word_count(turn.text) for turn in turns[start : end + 1])
            section["approx_word_count"] = size
            if size < 500 or size > 1800:
                warnings.append(f"SOFT:section length {size} words")
        expected = end + 1
    if expected != len(turns):
        warnings.append(f"HARD:coverage ends at {expected - 1}, expected {len(turns) - 1}")
    return warnings


def _needs_review(file_name: str, warnings: list[str], review_sample: int) -> bool:
    stable_sample = int.from_bytes(hashlib.sha256(file_name.encode("utf-8")).digest()[:4], "big") % 100 < review_sample
    return stable_sample or bool(warnings)


def _process_one(source_file: Path, output_file: Path, runtime: Path, agent: Any, review_sample: int) -> dict[str, Any]:
    fields, source_body = split_front_matter(source_file.read_bytes())
    body = source_body.decode("utf-8")
    turns = parse_turns(body)
    transcript = numbered_transcript(turns)
    plan, usage = agent.plan(file_name=source_file.name, metadata=fields, transcript=transcript)
    warnings = _plan_warnings(plan, turns)
    reviewed = False
    if _needs_review(source_file.name, warnings, review_sample):
        plan, review_usage = agent.plan(
            file_name=source_file.name,
            metadata=fields,
            transcript=transcript,
            review={"plan": plan, "warnings": warnings},
        )
        usage = {key: (usage.get(key) or 0) + (review_usage.get(key) or 0) for key in set(usage) | set(review_usage)}
        warnings = _plan_warnings(plan, turns)
        reviewed = True
    hard = [item for item in warnings if item.startswith("HARD:")]
    if hard:
        raise ValueError("; ".join(hard))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_file.parent, prefix=f".{output_file.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(document_heading("podcast", fields).encode("utf-8") + b"\n" + source_body)
        render_podcast(temporary_path, plan)
        validation = validate_document(source_file, temporary_path, require_chapters=True)
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))
        temporary_path.replace(output_file)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    plan_path = runtime / "plans" / f"{source_file.name}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _append_jsonl(runtime / "usage.jsonl", {"file": source_file.name, "reviewed": reviewed, **usage})
    _append_jsonl(runtime / "state.jsonl", {"file": source_file.name, "status": "done", "reviewed": reviewed, "warnings": warnings})
    return {"file": source_file.name, "status": "done", "reviewed": reviewed, "h2": validation["h2"], "h3": validation["h3"]}


def chapter_all(*, source: Path, output: Path, runtime: Path, agent: Any, concurrency: int, review_sample: int, resume: bool, only: list[str] | None) -> dict[str, int]:
    source_files = sorted((source / "03-podcasts").glob("*.md"))
    if only:
        wanted = set(only)
        source_files = [item for item in source_files if item.name in wanted]
    done: set[str] = set()
    state_file = runtime / "state.jsonl"
    if resume and state_file.exists():
        for line in state_file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("status") == "done":
                    done.add(str(row["file"]))
            except json.JSONDecodeError:
                continue
    source_files = [item for item in source_files if item.name not in done]
    result = {"requested": len(source_files), "done": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(_process_one, source_file, output / "03-podcasts" / source_file.name, runtime, agent, review_sample): source_file
            for source_file in source_files
        }
        for future in as_completed(futures):
            source_file = futures[future]
            try:
                future.result()
                result["done"] += 1
            except Exception as exc:
                result["failed"] += 1
                _append_jsonl(runtime / "failures.jsonl", {"file": source_file.name, "error": str(exc)})
    return result


def report_state(runtime: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runtime": str(runtime),
        "done": 0,
        "failure_records": 0,
        "resolved_failures": 0,
        "unresolved_failures": 0,
        "usage_records": 0,
    }
    if not runtime.exists():
        return result
    done_files: set[str] = set()
    state_path = runtime / "state.jsonl"
    if state_path.exists():
        for line in state_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("status") == "done":
                    done_files.add(str(row["file"]))
            except (json.JSONDecodeError, KeyError):
                continue
    failure_files: set[str] = set()
    failure_path = runtime / "failures.jsonl"
    if failure_path.exists():
        failure_lines = failure_path.read_text(encoding="utf-8").splitlines()
        result["failure_records"] = len(failure_lines)
        for line in failure_lines:
            try:
                failure_files.add(str(json.loads(line)["file"]))
            except (json.JSONDecodeError, KeyError):
                continue
    usage_path = runtime / "usage.jsonl"
    if usage_path.exists():
        result["usage_records"] = len(usage_path.read_text(encoding="utf-8").splitlines())
    result["done"] = len(done_files)
    result["resolved_failures"] = len(failure_files & done_files)
    result["unresolved_failures"] = len(failure_files - done_files)
    return result
