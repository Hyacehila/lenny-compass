from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .frontmatter import document_heading, markdown_headings, normalize_newsletter_body, split_front_matter
from .provenance import validate_enhanced_manifest, validate_index_metadata
from .transcript import parse_turns


DISCARDED_FIELD_RE = re.compile(r"^(?:type|word_count|description|channel|youtube_url|video_id):", re.M)
SOURCE_SNAPSHOT_NAME = "source-sha256.json"


def source_snapshot(source: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        data = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {"file_count": len(files), "files": files}


def write_source_snapshot(source: Path, runtime: Path) -> dict[str, Any]:
    snapshot = source_snapshot(source)
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / SOURCE_SNAPSHOT_NAME).write_bytes(
        (json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    return snapshot


def validate_source_snapshot(source: Path, runtime: Path) -> dict[str, Any]:
    path = runtime / SOURCE_SNAPSHOT_NAME
    if not path.exists():
        return {"valid": False, "error": "source SHA-256 baseline is missing"}
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = source_snapshot(source)
    return {
        "valid": expected == actual,
        "expected_files": expected.get("file_count"),
        "actual_files": actual["file_count"],
        "error": None if expected == actual else "archive source files changed after prepare",
    }


def validate_document(source: Path, enhanced: Path, require_chapters: bool) -> dict[str, Any]:
    source_fields, source_body = split_front_matter(source.read_bytes())
    enhanced_data = enhanced.read_bytes()
    enhanced_text = enhanced_data.decode("utf-8")
    headings = markdown_headings(enhanced_text)
    heading_levels = [level for level, _ in headings]
    errors: list[str] = []
    source_type = str(source_fields.get("type") or "")
    expected_heading = document_heading(source_type, source_fields).encode("utf-8") + b"\n"
    if not enhanced_data.startswith(expected_heading):
        errors.append("generated H1 is not the first line")
        generated_body_data = enhanced_data
    else:
        generated_body_data = enhanced_data[len(expected_heading):]
    if enhanced_text.startswith("---"):
        errors.append("Front Matter remains")
    if DISCARDED_FIELD_RE.search(enhanced_text):
        errors.append("discarded metadata field remains")
    if heading_levels.count(1) != 1:
        errors.append(f"expected exactly one semantic H1, found {heading_levels.count(1)}")
    if not headings or headings[0] != (1, 0):
        errors.append("generated H1 is not the first semantic heading")

    generated_body_text = generated_body_data.decode("utf-8")
    generated_body = generated_body_text.splitlines()
    expected_lines = [line for line in source_body.decode("utf-8").splitlines() if line.strip()]
    if source_type == "newsletter":
        expected_body, _ = normalize_newsletter_body(source_body, source.name)
        if generated_body_data != expected_body:
            errors.append("newsletter body differs from deterministic normalization")
    if source_type == "podcast":
        generated_body = [line for line in generated_body if not re.match(r"^#{2,3}\s+", line)]
        actual_lines = [line for line in generated_body if line.strip()]
        if expected_lines != actual_lines:
            errors.append("non-heading body differs")
    try:
        source_turns = parse_turns(source_body.decode("utf-8"))
        actual_turns = parse_turns("\n".join(generated_body))
        if len(source_turns) != len(actual_turns):
            errors.append("speaker turn count differs")
    except ValueError as exc:
        if source_fields.get("type") == "podcast":
            errors.append(str(exc))
    if require_chapters and (2 not in heading_levels or 3 not in heading_levels):
        errors.append("missing H2/H3 chapter structure")
    return {
        "file": str(enhanced),
        "valid": not errors,
        "errors": errors,
        "h1": heading_levels.count(1),
        "h2": heading_levels.count(2),
        "h3": heading_levels.count(3),
    }


def validate_newsletters(source: Path, output: Path) -> dict[str, Any]:
    source_files = sorted((source / "02-newsletters").glob("*.md"))
    output_files = sorted((output / "02-newsletters").glob("*.md"))
    if {item.name for item in source_files} != {item.name for item in output_files}:
        return {"valid": False, "documents": 0, "invalid": 1, "details": [{"errors": ["file set mismatch"]}]}
    results = [validate_document(path, output / "02-newsletters" / path.name, False) for path in source_files]
    return {
        "valid": bool(results) and all(item["valid"] for item in results),
        "documents": len(results),
        "invalid": sum(not item["valid"] for item in results),
        "details": [item for item in results if not item["valid"]],
    }


def validate_all(
    source: Path,
    output: Path,
    require_chapters: bool,
    runtime: Path | None = None,
    require_manifest: bool = False,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for folder in ("02-newsletters", "03-podcasts"):
        source_files = sorted((source / folder).glob("*.md"))
        output_files = sorted((output / folder).glob("*.md"))
        if {item.name for item in source_files} != {item.name for item in output_files}:
            results.append({"folder": folder, "valid": False, "errors": ["file set mismatch"]})
            continue
        for source_file in source_files:
            results.append(validate_document(source_file, output / folder / source_file.name, require_chapters and folder == "03-podcasts"))
    snapshot = validate_source_snapshot(source, runtime) if runtime is not None else {"valid": True}
    index_metadata = validate_index_metadata(source)
    if require_manifest and runtime is None:
        enhanced_manifest = {"valid": False, "documents": 0, "error": "provenance path is required"}
    elif require_manifest:
        enhanced_manifest = validate_enhanced_manifest(source, output, runtime)
    else:
        enhanced_manifest = {"valid": True, "required": False}
    return {
        "valid": (
            bool(results)
            and all(item["valid"] for item in results)
            and snapshot["valid"]
            and index_metadata["valid"]
            and enhanced_manifest["valid"]
        ),
        "documents": len(results),
        "invalid": sum(not item["valid"] for item in results),
        "source_snapshot": snapshot,
        "source_index_metadata": index_metadata,
        "enhanced_manifest": enhanced_manifest,
        "details": [item for item in results if not item["valid"]],
    }
