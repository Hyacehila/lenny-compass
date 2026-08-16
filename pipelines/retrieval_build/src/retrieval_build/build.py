from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ai_pm_contracts import FactBuildManifest, FactBuildManifestEntry, FactDocument
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .database import (
    check_semantic_runtime_versions,
    check_sqlite_version,
    connect_lexical,
    connect_semantic,
    create_lexical_schema,
    create_semantic_schema,
    get_index_status,
    initialize_index_meta,
    insert_lexical_document,
    insert_semantic_segments,
    insert_vectors,
    optimize_and_check_fts,
    set_meta,
)
from .embeddings import EMBEDDING_DIMENSIONS, EmbeddingProvider
from .errors import IndexBuildError
from .semantic_segments import embed_with_length_fallback


class RetrievalBundleManifest(BaseModel):
    """Commit record binding both standalone indexes to one facts snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1"] = "1"
    build_id: str = Field(min_length=1)
    created_at: datetime
    facts_manifest_file: str = Field(min_length=1)
    facts_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)

    lexical_status: Literal["ready"] = "ready"
    lexical_database_file: str = Field(min_length=1)
    lexical_database_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    semantic_status: Literal["ready", "not_ready", "test_only"]
    semantic_reason: str | None = None
    semantic_database_file: str | None = None
    semantic_database_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    embedding_model: str = Field(min_length=1)
    embedding_dimensions: Literal[2048] = 2048

    sqlite_version: str = Field(min_length=1)
    sqlite_vec_version: str | None = None

    @model_validator(mode="after")
    def validate_semantic_artifact(self) -> RetrievalBundleManifest:
        has_database = self.semantic_database_file is not None
        has_hash = self.semantic_database_sha256 is not None
        if self.semantic_status == "not_ready":
            if has_database or has_hash or self.sqlite_vec_version is not None:
                raise ValueError(
                    "not_ready semantic status must not publish an artifact"
                )
        elif not (has_database and has_hash and self.sqlite_vec_version):
            raise ValueError(
                "ready/test_only semantic status requires a complete artifact"
            )
        return self


def build_retrieval_bundle(
    *,
    facts_manifest_path: str | Path,
    lexical_output_path: str | Path,
    semantic_output_path: str | Path,
    manifest_output_path: str | Path,
    embedder: EmbeddingProvider | None = None,
    semantic_reason: str | None = None,
) -> RetrievalBundleManifest:
    """Build separate lexical/vector databases and publish the manifest last.

    The bundle manifest is the commit marker. If a process stops between a
    database replacement and the manifest replacement, startup hash/build-id
    validation fails closed instead of mixing generations.
    """

    facts_manifest_file = Path(facts_manifest_path).resolve(strict=True)
    # Contract: facts/manifests/build.json points to paths relative to facts/.
    if facts_manifest_file.parent.name.casefold() != "manifests":
        raise IndexBuildError(
            "INVALID_FACT_MANIFEST_PATH",
            "facts manifest must be inside the facts/manifests directory",
        )
    facts_root = facts_manifest_file.parent.parent
    facts_manifest_bytes = facts_manifest_file.read_bytes()
    facts_manifest_sha256 = hashlib.sha256(facts_manifest_bytes).hexdigest()
    try:
        facts_manifest = FactBuildManifest.model_validate_json(facts_manifest_bytes)
    except ValueError as exc:
        raise IndexBuildError("INVALID_FACT_MANIFEST", str(exc)) from exc

    manifest_output = Path(manifest_output_path).resolve(strict=False)
    # Contract: retrieval/manifests/build.json points to paths relative to retrieval/.
    if manifest_output.parent.name.casefold() != "manifests":
        raise IndexBuildError(
            "INVALID_OUTPUT_PATH",
            "bundle manifest must be inside the retrieval/manifests directory",
        )
    bundle_root = manifest_output.parent.parent
    repository_root = bundle_root.parent
    try:
        facts_manifest_file.relative_to(repository_root)
    except ValueError as exc:
        raise IndexBuildError(
            "INVALID_FACT_MANIFEST_PATH",
            "facts and retrieval roots must share the same repository root",
        ) from exc
    lexical_output = _validated_output_path(bundle_root, lexical_output_path, "lexical")
    semantic_output = _validated_output_path(
        bundle_root, semantic_output_path, "semantic"
    )
    if lexical_output == semantic_output:
        raise IndexBuildError(
            "INVALID_OUTPUT_PATH",
            "lexical and semantic outputs must be different files",
        )
    if manifest_output in {lexical_output, semantic_output}:
        raise IndexBuildError(
            "INVALID_OUTPUT_PATH",
            "bundle manifest must be separate from database files",
        )

    bundle_root.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    lexical_output.parent.mkdir(parents=True, exist_ok=True)
    semantic_output.parent.mkdir(parents=True, exist_ok=True)

    build_id = uuid.uuid4().hex
    lexical_staging = lexical_output.with_name(f".{lexical_output.name}.{build_id}.tmp")
    semantic_staging = semantic_output.with_name(
        f".{semantic_output.name}.{build_id}.tmp"
    )
    manifest_staging = manifest_output.with_name(
        f".{manifest_output.name}.{build_id}.tmp"
    )

    lexical_connection: sqlite3.Connection | None = None
    semantic_connection: sqlite3.Connection | None = None
    try:
        lexical_connection = connect_lexical(lexical_staging)
        sqlite_version = check_sqlite_version(lexical_connection)
        create_lexical_schema(lexical_connection)
        initialize_index_meta(
            lexical_connection,
            index_kind="lexical",
            build_id=build_id,
            facts_manifest_sha256=facts_manifest_sha256,
        )

        sqlite_vec_version: str | None = None
        if embedder is not None:
            if embedder.dimensions != EMBEDDING_DIMENSIONS:
                raise IndexBuildError(
                    "EMBEDDING_DIMENSION_MISMATCH",
                    f"embedder declares {embedder.dimensions} dimensions; expected {EMBEDDING_DIMENSIONS}",
                )
            semantic_connection = connect_semantic(semantic_staging)
            _, sqlite_vec_version = check_semantic_runtime_versions(semantic_connection)
            create_semantic_schema(semantic_connection)
            initialize_index_meta(
                semantic_connection,
                index_kind="semantic",
                build_id=build_id,
                facts_manifest_sha256=facts_manifest_sha256,
                embedding_model=embedder.model_name,
            )

        loaded_documents = 0
        loaded_chunks = 0
        for entry in facts_manifest.documents:
            document_path = _resolve_under_root(facts_root, entry.fact_document_path)
            try:
                fact_bytes = _read_and_verify_sha256(
                    document_path,
                    entry.fact_document_sha256,
                    label="fact document",
                )
                _read_and_verify_sha256(
                    _resolve_under_root(repository_root, entry.raw_path),
                    entry.raw_sha256,
                    label="raw source",
                )
                _read_and_verify_sha256(
                    _resolve_under_root(repository_root, entry.enhanced_path),
                    entry.enhanced_sha256,
                    label="enhanced source",
                )
                _read_and_verify_sha256(
                    _resolve_under_root(facts_root, entry.docling_path),
                    entry.docling_sha256,
                    label="Docling document",
                )
                document = FactDocument.model_validate_json(fact_bytes)
            except (OSError, ValueError) as exc:
                raise IndexBuildError(
                    "INVALID_FACT_DOCUMENT",
                    f"failed to load {entry.fact_document_path}: {exc}",
                ) from exc
            _validate_manifest_entry(document, entry)
            ordered_chunks = sorted(document.chunks, key=lambda item: item.ordinal)
            insert_lexical_document(lexical_connection, document)
            if (
                semantic_connection is not None
                and embedder is not None
                and ordered_chunks
            ):
                embedded_chunks = embed_with_length_fallback(
                    embedder, [chunk.contextualized_text for chunk in ordered_chunks]
                )
                semantic_segments = []
                embeddings = []
                for chunk, chunk_segments in zip(
                    ordered_chunks, embedded_chunks, strict=True
                ):
                    for embedding_segment_ordinal, segment in enumerate(chunk_segments):
                        semantic_segments.append(
                            (chunk.locator, embedding_segment_ordinal)
                        )
                        embeddings.append(segment.embedding)
                rowids = insert_semantic_segments(
                    semantic_connection, semantic_segments
                )
                insert_vectors(semantic_connection, rowids, embeddings)
            loaded_documents += 1
            loaded_chunks += len(ordered_chunks)

        if loaded_documents != facts_manifest.document_count:
            raise IndexBuildError(
                "FACT_COUNT_MISMATCH", "loaded document count does not match manifest"
            )
        if loaded_chunks != facts_manifest.chunk_count:
            raise IndexBuildError(
                "FACT_COUNT_MISMATCH", "loaded chunk count does not match manifest"
            )

        optimize_and_check_fts(lexical_connection)
        set_meta(lexical_connection, "status", "ready")
        lexical_integrity = str(
            lexical_connection.execute("PRAGMA integrity_check").fetchone()[0]
        )
        if lexical_integrity != "ok":
            raise IndexBuildError("SQLITE_INTEGRITY_ERROR", lexical_integrity)
        lexical_connection.commit()
        lexical_connection.close()
        lexical_connection = None

        semantic_status = "not_ready"
        if semantic_connection is not None and embedder is not None:
            semantic_status = embedder.status
            if semantic_status not in {"ready", "test_only"}:
                raise IndexBuildError(
                    "INVALID_EMBEDDER_STATUS",
                    f"embedder reported unsupported status: {semantic_status}",
                )
            set_meta(semantic_connection, "status", semantic_status)
            semantic_integrity = str(
                semantic_connection.execute("PRAGMA integrity_check").fetchone()[0]
            )
            if semantic_integrity != "ok":
                raise IndexBuildError("SQLITE_INTEGRITY_ERROR", semantic_integrity)
            semantic_connection.commit()
            semantic_connection.close()
            semantic_connection = None

        lexical_sha256 = _sha256_file(lexical_staging)
        semantic_sha256 = (
            _sha256_file(semantic_staging) if semantic_status != "not_ready" else None
        )
        manifest = RetrievalBundleManifest(
            build_id=build_id,
            created_at=datetime.now(UTC),
            facts_manifest_file=_relative_artifact_path(
                repository_root, facts_manifest_file
            ),
            facts_manifest_sha256=facts_manifest_sha256,
            document_count=loaded_documents,
            chunk_count=loaded_chunks,
            lexical_database_file=_relative_artifact_path(bundle_root, lexical_output),
            lexical_database_sha256=lexical_sha256,
            semantic_status=semantic_status,
            semantic_reason=(
                semantic_reason
                or (
                    "semantic embedder was not configured" if embedder is None else None
                )
            ),
            semantic_database_file=(
                _relative_artifact_path(bundle_root, semantic_output)
                if semantic_status != "not_ready"
                else None
            ),
            semantic_database_sha256=semantic_sha256,
            embedding_model=embedder.model_name if embedder else "embedding-3",
            sqlite_version=sqlite_version,
            sqlite_vec_version=sqlite_vec_version,
        )
        _write_json_fsync(manifest_staging, manifest.model_dump(mode="json"))

        # Publish databases first and the manifest last as the atomic commit marker.
        os.replace(lexical_staging, lexical_output)
        if semantic_status != "not_ready":
            os.replace(semantic_staging, semantic_output)
        else:
            # A prior ready generation must never look usable beside a new
            # not-ready commit marker. The manifest is still published last.
            semantic_output.unlink(missing_ok=True)
        os.replace(manifest_staging, manifest_output)
        return manifest
    except Exception:
        if lexical_connection is not None:
            lexical_connection.close()
        if semantic_connection is not None:
            semantic_connection.close()
        lexical_staging.unlink(missing_ok=True)
        semantic_staging.unlink(missing_ok=True)
        manifest_staging.unlink(missing_ok=True)
        raise


def load_and_validate_bundle_manifest(
    manifest_path: str | Path,
) -> RetrievalBundleManifest:
    """Validate artifact hashes, build IDs, facts hash, and runtime versions."""

    manifest_file = Path(manifest_path).resolve(strict=True)
    # Contract: retrieval/manifests/build.json points to paths relative to retrieval/.
    if manifest_file.parent.name.casefold() != "manifests":
        raise IndexBuildError(
            "INVALID_BUILD_MANIFEST",
            "bundle manifest must be inside the retrieval/manifests directory",
        )
    bundle_root = manifest_file.parent.parent
    try:
        manifest = RetrievalBundleManifest.model_validate_json(
            manifest_file.read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise IndexBuildError("INVALID_BUILD_MANIFEST", str(exc)) from exc

    facts_manifest = _resolve_under_root(
        bundle_root.parent,
        manifest.facts_manifest_file,
        code="INVALID_BUILD_MANIFEST",
    )
    if _sha256_file(facts_manifest) != manifest.facts_manifest_sha256:
        raise IndexBuildError("STALE_BUILD_MANIFEST", "facts manifest hash changed")

    lexical_database = _resolve_under_root(
        bundle_root, manifest.lexical_database_file, code="INVALID_BUILD_MANIFEST"
    )
    if _sha256_file(lexical_database) != manifest.lexical_database_sha256:
        raise IndexBuildError("STALE_BUILD_MANIFEST", "lexical database hash changed")
    with connect_lexical(lexical_database, query_only=True) as connection:
        status = get_index_status(connection)
        _validate_index_identity(
            index_kind="lexical",
            build_id=manifest.build_id,
            facts_manifest_sha256=manifest.facts_manifest_sha256,
            expected_status="ready",
            actual=status,
        )

    if manifest.semantic_status != "not_ready":
        if not manifest.semantic_database_file or not manifest.semantic_database_sha256:
            raise IndexBuildError(
                "INVALID_BUILD_MANIFEST", "semantic artifact metadata is incomplete"
            )
        semantic_database = _resolve_under_root(
            bundle_root,
            manifest.semantic_database_file,
            code="INVALID_BUILD_MANIFEST",
        )
        if _sha256_file(semantic_database) != manifest.semantic_database_sha256:
            raise IndexBuildError(
                "STALE_BUILD_MANIFEST", "semantic database hash changed"
            )
        with connect_semantic(semantic_database, query_only=True) as connection:
            status = get_index_status(connection)
            _validate_index_identity(
                index_kind="semantic",
                build_id=manifest.build_id,
                facts_manifest_sha256=manifest.facts_manifest_sha256,
                expected_status=manifest.semantic_status,
                actual=status,
            )
            _, vec_version = check_semantic_runtime_versions(connection)
            if vec_version != manifest.sqlite_vec_version:
                raise IndexBuildError(
                    "STALE_BUILD_MANIFEST", "sqlite-vec runtime version changed"
                )
    return manifest


def _validate_index_identity(
    *,
    index_kind: str,
    build_id: str,
    facts_manifest_sha256: str,
    expected_status: str,
    actual: object,
) -> None:
    values = {
        "index_kind": getattr(actual, "index_kind", None),
        "build_id": getattr(actual, "build_id", None),
        "facts_manifest_sha256": getattr(actual, "facts_manifest_sha256", None),
        "status": getattr(actual, "status", None),
    }
    expected = {
        "index_kind": index_kind,
        "build_id": build_id,
        "facts_manifest_sha256": facts_manifest_sha256,
        "status": expected_status,
    }
    if values != expected:
        raise IndexBuildError(
            "STALE_BUILD_MANIFEST",
            f"index identity does not match bundle manifest: {values!r}",
        )


def _validate_manifest_entry(
    document: FactDocument, entry: FactBuildManifestEntry
) -> None:
    if document.document_id != entry.document_id:
        raise IndexBuildError(
            "FACT_MANIFEST_MISMATCH", "document_id does not match manifest entry"
        )
    if document.raw_path != entry.raw_path:
        raise IndexBuildError(
            "FACT_MANIFEST_MISMATCH", "raw_path does not match manifest entry"
        )
    if document.raw_sha256 != entry.raw_sha256:
        raise IndexBuildError(
            "FACT_MANIFEST_MISMATCH", "raw hash does not match manifest entry"
        )
    if document.enhanced_path != entry.enhanced_path:
        raise IndexBuildError(
            "FACT_MANIFEST_MISMATCH", "enhanced_path does not match manifest entry"
        )
    if document.enhanced_sha256 != entry.enhanced_sha256:
        raise IndexBuildError(
            "FACT_MANIFEST_MISMATCH", "enhanced hash does not match manifest entry"
        )
    if document.docling_path != entry.docling_path:
        raise IndexBuildError(
            "FACT_MANIFEST_MISMATCH", "Docling path does not match manifest entry"
        )
    if document.docling_sha256 != entry.docling_sha256:
        raise IndexBuildError(
            "FACT_MANIFEST_MISMATCH", "Docling hash does not match manifest entry"
        )
    if document.fact_document_path != entry.fact_document_path:
        raise IndexBuildError(
            "FACT_MANIFEST_MISMATCH", "fact path does not match manifest entry"
        )
    if len(document.chunks) != entry.chunk_count:
        raise IndexBuildError(
            "FACT_MANIFEST_MISMATCH", "chunk count does not match manifest entry"
        )


def _validated_output_path(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise IndexBuildError(
            "INVALID_OUTPUT_PATH",
            f"{label} output must be beneath the bundle manifest directory",
        ) from exc
    return path


def _relative_artifact_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _resolve_under_root(
    root: Path, relative_path: str, *, code: str = "INVALID_FACT_PATH"
) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise IndexBuildError(code, "absolute artifact paths are not allowed")
    candidate = (root / path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise IndexBuildError(
            code, "artifact path escapes its configured root"
        ) from exc
    return candidate


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise IndexBuildError("MISSING_INDEX_ARTIFACT", str(exc)) from exc


def _read_and_verify_sha256(path: Path, expected: str, *, label: str) -> bytes:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise IndexBuildError(
            "FACT_ARTIFACT_HASH_MISMATCH",
            f"{label} hash changed: {path}",
        )
    return payload


def _write_json_fsync(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
