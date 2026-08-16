from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__
from .deepseek_agent import PLAN_SCHEMA, SYSTEM_PROMPT
from .frontmatter import split_front_matter


ENHANCED_MANIFEST_NAME = "enhanced-manifest.json"
RUN_SUMMARY_NAME = "run-summary.json"
SOURCE_SNAPSHOT_NAME = "source-sha256.json"
DOCUMENT_FOLDERS = ("02-newsletters", "03-podcasts")
INDEX_FIELDS = {
    "newsletter": ("title", "subtitle", "date", "tags", "word_count"),
    "podcast": ("title", "date", "guest", "description", "tags", "word_count"),
}
TRANSFORMATION_CODE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "src/archive_enhancer/__init__.py",
    "src/archive_enhancer/deepseek_agent.py",
    "src/archive_enhancer/frontmatter.py",
    "src/archive_enhancer/pipeline.py",
    "src/archive_enhancer/renderer.py",
    "src/archive_enhancer/transcript.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def normalize_metadata_for_comparison(value: Any) -> Any:
    """Normalize comparison-only whitespace without mutating authoritative metadata."""
    if isinstance(value, str):
        return value.replace("\u00a0", " ")
    if isinstance(value, list):
        return [normalize_metadata_for_comparison(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_metadata_for_comparison(item) for key, item in value.items()}
    return value


def metadata_values_equal(left: Any, right: Any) -> bool:
    return normalize_metadata_for_comparison(left) == normalize_metadata_for_comparison(right)


def _safe_index_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
        return None
    return path.as_posix()


def validate_index_metadata(source: Path) -> dict[str, Any]:
    index_path = source / "01-start-here" / "index.json"
    if not index_path.is_file():
        return {"valid": False, "documents": 0, "differences": [{"error": "source index is missing"}]}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"valid": False, "documents": 0, "differences": [{"error": f"invalid source index: {exc}"}]}

    differences: list[dict[str, Any]] = []
    indexed_paths: set[str] = set()
    expected_paths = {
        path.relative_to(source).as_posix()
        for folder in DOCUMENT_FOLDERS
        for path in (source / folder).glob("*.md")
    }
    groups = (("newsletters", "newsletter"), ("podcasts", "podcast"))
    for group_name, expected_type in groups:
        items = index.get(group_name)
        if not isinstance(items, list):
            differences.append({"group": group_name, "error": "index group is not a list"})
            continue
        for item in items:
            if not isinstance(item, dict):
                differences.append({"group": group_name, "error": "index item is not an object"})
                continue
            relative = _safe_index_path(item.get("filename"))
            if relative is None:
                differences.append({"group": group_name, "filename": item.get("filename"), "error": "unsafe filename"})
                continue
            if relative in indexed_paths:
                differences.append({"filename": relative, "error": "duplicate filename"})
                continue
            indexed_paths.add(relative)
            raw_path = source / PurePosixPath(relative)
            if not raw_path.is_file():
                differences.append({"filename": relative, "error": "indexed file is missing"})
                continue
            try:
                fields, _ = split_front_matter(raw_path.read_bytes())
            except (UnicodeDecodeError, ValueError) as exc:
                differences.append({"filename": relative, "error": f"invalid front matter: {exc}"})
                continue
            if fields.get("type") != expected_type:
                differences.append(
                    {
                        "filename": relative,
                        "field": "type",
                        "frontmatter": fields.get("type"),
                        "index_group": expected_type,
                    }
                )
            for field in INDEX_FIELDS[expected_type]:
                if not metadata_values_equal(fields.get(field), item.get(field)):
                    differences.append(
                        {
                            "filename": relative,
                            "field": field,
                            "frontmatter": fields.get(field),
                            "index": item.get(field),
                        }
                    )

    for missing in sorted(expected_paths - indexed_paths):
        differences.append({"filename": missing, "error": "document is not indexed"})
    for extra in sorted(indexed_paths - expected_paths):
        differences.append({"filename": extra, "error": "index entry has no corpus document"})
    return {
        "valid": not differences,
        "documents": len(indexed_paths),
        "comparison_normalization": "replace U+00A0 NBSP with U+0020 SPACE for equality only",
        "differences": differences,
    }


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": _display_path(path, repo_root),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def transformation_code_snapshot() -> dict[str, Any]:
    pipeline_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for relative in TRANSFORMATION_CODE_FILES:
        path = pipeline_root / PurePosixPath(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        file_hash = sha256_bytes(data)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        files.append({"path": relative, "bytes": len(data), "sha256": file_hash})
    return {
        "version": __version__,
        "tree_sha256": digest.hexdigest(),
        "tree_hash_scheme": "sha256(path\\0size\\0file_sha256\\n)",
        "files": files,
    }


def build_enhanced_manifest(source: Path, output: Path, provenance: Path) -> dict[str, Any]:
    repo_root = source.parents[1]
    run_summary_path = provenance / RUN_SUMMARY_NAME
    source_snapshot_path = provenance / SOURCE_SNAPSHOT_NAME
    if not run_summary_path.is_file():
        raise FileNotFoundError(run_summary_path)
    if not source_snapshot_path.is_file():
        raise FileNotFoundError(source_snapshot_path)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))

    documents: list[dict[str, Any]] = []
    for folder in DOCUMENT_FOLDERS:
        raw_files = {path.name: path for path in (source / folder).glob("*.md")}
        enhanced_files = {path.name: path for path in (output / folder).glob("*.md")}
        if raw_files.keys() != enhanced_files.keys():
            raise ValueError(f"raw/enhanced file set mismatch in {folder}")
        for name in sorted(raw_files):
            raw_path = raw_files[name]
            enhanced_path = enhanced_files[name]
            fields, _ = split_front_matter(raw_path.read_bytes())
            expected_type = "newsletter" if folder == "02-newsletters" else "podcast"
            if fields.get("type") != expected_type:
                raise ValueError(f"{raw_path}: expected type={expected_type!r}")
            documents.append(
                {
                    "document_id": f"{folder}/{name}",
                    "document_type": expected_type,
                    "metadata_authority": "raw_frontmatter",
                    "metadata": fields,
                    "raw": _file_record(raw_path, repo_root),
                    "enhanced": _file_record(enhanced_path, repo_root),
                    "transformation": (
                        "deterministic_newsletter_heading_normalization_v1"
                        if expected_type == "newsletter"
                        else "deepseek_podcast_chapter_insertion_v1"
                    ),
                }
            )

    index_validation = validate_index_metadata(source)
    if not index_validation["valid"]:
        raise ValueError(f"source index metadata mismatch: {index_validation['differences'][:3]}")
    return {
        "schema_version": 1,
        "manifest_type": "enhanced_corpus_provenance",
        "metadata_policy": {
            "authority": "YAML front matter in corpus/raw Markdown",
            "index": "corpus/raw/01-start-here/index.json",
            "comparison_normalization": index_validation["comparison_normalization"],
            "stored_values": "raw front matter is stored without normalization",
        },
        "generation_run": {
            "run_summary": _file_record(run_summary_path, repo_root),
            "source_snapshot": _file_record(source_snapshot_path, repo_root),
            "run_date": run_summary.get("run_date"),
            "raw_tree_sha256": (run_summary.get("source") or {}).get("tree_sha256"),
            "enhanced_tree_sha256": (run_summary.get("output") or {}).get("tree_sha256"),
        },
        "pipeline": {
            "name": "archive_enhancer",
            "transformation_code": transformation_code_snapshot(),
            "newsletter": {
                "mode": "deterministic",
                "strategy": "demote every semantic body heading by one when the body contains H1; convert H6 to bold",
            },
            "podcast_chaptering": {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "model_evidence": "archive_enhancer CLI default; the historical run summary did not persist per-request model names",
                "temperature": 0.2,
                "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
                "plan_schema_sha256": _json_sha256(PLAN_SCHEMA),
                "plans_retained": False,
            },
        },
        "document_count": len(documents),
        "documents": documents,
    }


def write_enhanced_manifest(source: Path, output: Path, provenance: Path) -> dict[str, Any]:
    manifest = build_enhanced_manifest(source, output, provenance)
    provenance.mkdir(parents=True, exist_ok=True)
    path = provenance / ENHANCED_MANIFEST_NAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    temporary.replace(path)
    return manifest


def validate_enhanced_manifest(source: Path, output: Path, provenance: Path) -> dict[str, Any]:
    path = provenance / ENHANCED_MANIFEST_NAME
    if not path.is_file():
        return {"valid": False, "documents": 0, "error": "enhanced provenance manifest is missing"}
    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
        actual = build_enhanced_manifest(source, output, provenance)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {"valid": False, "documents": 0, "error": str(exc)}
    return {
        "valid": expected == actual,
        "documents": actual["document_count"],
        "error": None if expected == actual else "enhanced provenance manifest does not match the corpus",
    }
