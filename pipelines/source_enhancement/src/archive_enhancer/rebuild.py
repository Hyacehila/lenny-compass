from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .frontmatter import build_newsletter_copy
from .provenance import write_enhanced_manifest
from .validator import validate_all, validate_newsletters, validate_source_snapshot


RUN_SUMMARY_NAME = "run-summary.json"


def directory_summary(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        file_hash = hashlib.sha256(data).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        byte_count += len(data)
    return {
        "file_count": file_count,
        "byte_count": byte_count,
        "tree_sha256": digest.hexdigest(),
        "tree_hash_scheme": "sha256(path\\0size\\0file_sha256\\n)",
    }


def _write_run_summary(
    provenance: Path,
    source: Path,
    output: Path,
    normalization: dict[str, Any],
    podcast_summary: dict[str, Any],
) -> None:
    path = provenance / RUN_SUMMARY_NAME
    repo_root = source.parents[1]

    def display_path(value: Path) -> str:
        try:
            return value.relative_to(repo_root).as_posix()
        except ValueError:
            return value.as_posix()

    summary: dict[str, Any] = {}
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema_version": 2,
            "pipeline": "archive_enhancer",
            "run_date": datetime.now(timezone.utc).isoformat(),
            "source": {
                "path": display_path(source),
                **directory_summary(source),
                "snapshot": "source-sha256.json",
            },
            "output": {
                "path": display_path(output),
                **directory_summary(output),
                "newsletter_documents": len(list((output / "02-newsletters").glob("*.md"))),
                "podcast_documents": len(list((output / "03-podcasts").glob("*.md"))),
            },
            "newsletter_normalization": {
                "strategy": "demote every semantic body heading by one when the body contains H1",
                "h6_strategy": "convert to bold paragraph",
                **normalization,
            },
            "integrity": {
                "source_snapshot_valid": True,
                "podcast_directory_unchanged": True,
                "podcast_tree_sha256": podcast_summary["tree_sha256"],
            },
        }
    )
    provenance.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    temporary.replace(path)


def rebuild_newsletters(source: Path, output: Path, runtime: Path, provenance: Path) -> dict[str, Any]:
    source_integrity = validate_source_snapshot(source, provenance)
    if not source_integrity["valid"]:
        raise ValueError(source_integrity.get("error") or "source SHA-256 validation failed")

    target = output / "02-newsletters"
    podcasts = output / "03-podcasts"
    if not target.is_dir():
        raise FileNotFoundError(target)
    if not podcasts.is_dir():
        raise FileNotFoundError(podcasts)

    podcast_before = directory_summary(podcasts)
    runtime.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="newsletter-rebuild-", dir=runtime))
    staging_root = temporary_root / "staging"
    staging_newsletters = staging_root / "02-newsletters"
    backup = temporary_root / "02-newsletters.backup"
    failed_output = temporary_root / "02-newsletters.failed"
    old_moved = False
    new_installed = False
    try:
        normalization = build_newsletter_copy(source, staging_root)
        staged_validation = validate_newsletters(source, staging_root)
        if not staged_validation["valid"]:
            raise ValueError(f"staged newsletter validation failed: {staged_validation['details'][:3]}")

        target.rename(backup)
        old_moved = True
        staging_newsletters.rename(target)
        new_installed = True

        final_validation = validate_all(source, output, require_chapters=True, runtime=provenance)
        if not final_validation["valid"]:
            raise ValueError(f"final enhanced-layer validation failed: {final_validation['details'][:3]}")
        podcast_after = directory_summary(podcasts)
        if podcast_after != podcast_before:
            raise ValueError("podcast directory changed during newsletter rebuild")

        _write_run_summary(provenance, source, output, normalization, podcast_after)
        enhanced_manifest = write_enhanced_manifest(source, output, provenance)
        final_validation = validate_all(
            source,
            output,
            require_chapters=True,
            runtime=provenance,
            require_manifest=True,
        )
        if not final_validation["valid"]:
            raise ValueError("final enhanced-layer provenance validation failed")
        shutil.rmtree(backup)
        old_moved = False
        return {
            **normalization,
            "staged_validation": staged_validation,
            "final_validation": final_validation,
            "source_snapshot": source_integrity,
            "podcast_directory": podcast_after,
            "enhanced_manifest_documents": enhanced_manifest["document_count"],
        }
    except Exception:
        if new_installed and target.exists():
            target.rename(failed_output)
        if old_moved and backup.exists():
            backup.rename(target)
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
