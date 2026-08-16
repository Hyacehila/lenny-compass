from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import sqlite_vec
from ai_pm_contracts import (
    DocumentChunkLocator,
    FactDocument,
    QueryRewrite,
    RetrievalHit,
    RetrievalIndexStatus,
)
from pydantic import JsonValue
from sqlite_vec import serialize_float32

from .embeddings import EMBEDDING_DIMENSIONS, validate_embeddings
from .errors import IndexBuildError, IndexNotReadyError
from .query import compile_fts_match

EXPECTED_SQLITE_VEC_VERSION = "0.1.9"
MINIMUM_SQLITE_VERSION = (3, 41, 0)
MAX_CANDIDATES = 50


def connect_lexical(
    path: str | Path, *, query_only: bool = False
) -> sqlite3.Connection:
    """Open the standalone FTS5 database without loading sqlite-vec."""

    connection = _connect(path)
    try:
        check_sqlite_version(connection)
        check_fts5_available(connection)
    except Exception:
        connection.close()
        raise
    if query_only:
        connection.execute("PRAGMA query_only = ON")
    return connection


def connect_semantic(
    path: str | Path, *, query_only: bool = False
) -> sqlite3.Connection:
    """Open the standalone vector database and verify sqlite-vec exactly."""

    connection = _connect(path)
    try:
        connection.enable_load_extension(True)
        try:
            sqlite_vec.load(connection)
        finally:
            connection.enable_load_extension(False)
        check_semantic_runtime_versions(connection)
    except Exception:
        connection.close()
        raise
    if query_only:
        connection.execute("PRAGMA query_only = ON")
    return connection


def _connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def check_sqlite_version(connection: sqlite3.Connection) -> str:
    sqlite_version = str(connection.execute("SELECT sqlite_version()").fetchone()[0])
    try:
        parsed = tuple(int(part) for part in sqlite_version.split(".")[:3])
    except ValueError as exc:
        raise IndexBuildError(
            "UNSUPPORTED_SQLITE_VERSION",
            f"could not parse SQLite version: {sqlite_version}",
        ) from exc
    if parsed < MINIMUM_SQLITE_VERSION:
        raise IndexBuildError(
            "UNSUPPORTED_SQLITE_VERSION",
            f"SQLite >= {'.'.join(map(str, MINIMUM_SQLITE_VERSION))} is required; found {sqlite_version}",
        )
    return sqlite_version


def check_semantic_runtime_versions(connection: sqlite3.Connection) -> tuple[str, str]:
    sqlite_version = check_sqlite_version(connection)
    vec_version = str(connection.execute("SELECT vec_version()").fetchone()[0])
    if vec_version.removeprefix("v") != EXPECTED_SQLITE_VEC_VERSION:
        raise IndexBuildError(
            "UNSUPPORTED_SQLITE_VEC_VERSION",
            f"sqlite-vec {EXPECTED_SQLITE_VEC_VERSION} is required; found {vec_version}",
        )
    return sqlite_version, vec_version


def check_fts5_available(connection: sqlite3.Connection) -> None:
    """Probe actual FTS5 virtual-table support, not just compile metadata."""

    try:
        connection.execute(
            "CREATE VIRTUAL TABLE temp.__retrieval_fts5_probe USING fts5(value)"
        )
        connection.execute("DROP TABLE temp.__retrieval_fts5_probe")
    except sqlite3.OperationalError as exc:
        raise IndexBuildError(
            "FTS5_UNAVAILABLE", "SQLite runtime does not provide FTS5"
        ) from exc


def create_lexical_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE retrieval_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) STRICT;

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            document_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            title TEXT NOT NULL,
            headings TEXT NOT NULL,
            metadata TEXT NOT NULL,
            body TEXT NOT NULL,
            locator_json TEXT NOT NULL,
            UNIQUE(document_id, ordinal)
        ) STRICT;

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            title,
            headings,
            metadata,
            body,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );

        CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, title, headings, metadata, body)
            VALUES (new.id, new.title, new.headings, new.metadata, new.body);
        END;

        CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, title, headings, metadata, body)
            VALUES ('delete', old.id, old.title, old.headings, old.metadata, old.body);
        END;

        CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, title, headings, metadata, body)
            VALUES ('delete', old.id, old.title, old.headings, old.metadata, old.body);
            INSERT INTO chunks_fts(rowid, title, headings, metadata, body)
            VALUES (new.id, new.title, new.headings, new.metadata, new.body);
        END;
        """
    )


def create_semantic_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE retrieval_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) STRICT;

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            embedding_segment_ordinal INTEGER NOT NULL
                CHECK (embedding_segment_ordinal >= 0),
            locator_json TEXT NOT NULL,
            UNIQUE(chunk_id, embedding_segment_ordinal),
            UNIQUE(document_id, ordinal, embedding_segment_ordinal)
        ) STRICT;

        CREATE VIRTUAL TABLE chunk_vectors USING vec0(
            chunk_rowid INTEGER PRIMARY KEY,
            embedding float[2048] distance_metric=cosine
        );
        """
    )


def set_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO retrieval_meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_meta(connection: sqlite3.Connection, key: str) -> str | None:
    try:
        row = connection.execute(
            "SELECT value FROM retrieval_meta WHERE key = ?", (key,)
        ).fetchone()
    except sqlite3.OperationalError as exc:
        raise IndexNotReadyError(
            "INVALID_INDEX", "retrieval_meta table is missing"
        ) from exc
    return None if row is None else str(row[0])


def get_index_status(connection: sqlite3.Connection) -> RetrievalIndexStatus:
    """Read and validate the identity/readiness of one standalone index."""

    required_keys = (
        "index_kind",
        "schema_version",
        "build_id",
        "facts_manifest_sha256",
        "status",
    )
    values = {key: get_meta(connection, key) for key in required_keys}
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise IndexNotReadyError(
            "INVALID_INDEX",
            f"retrieval metadata is missing: {', '.join(missing)}",
        )
    dimensions = get_meta(connection, "embedding_dimensions")
    try:
        return RetrievalIndexStatus(
            index_kind=values["index_kind"],
            schema_version=values["schema_version"],
            build_id=values["build_id"],
            facts_manifest_sha256=values["facts_manifest_sha256"],
            status=values["status"],
            embedding_model=get_meta(connection, "embedding_model"),
            embedding_dimensions=int(dimensions) if dimensions is not None else None,
        )
    except (TypeError, ValueError) as exc:
        raise IndexNotReadyError(
            "INVALID_INDEX",
            f"retrieval metadata is invalid: {exc}",
        ) from exc


def initialize_index_meta(
    connection: sqlite3.Connection,
    *,
    index_kind: str,
    build_id: str,
    facts_manifest_sha256: str,
    status: str = "building",
    embedding_model: str | None = None,
) -> None:
    values = {
        "index_kind": index_kind,
        "schema_version": "1",
        "build_id": build_id,
        "facts_manifest_sha256": facts_manifest_sha256,
        "status": status,
    }
    if embedding_model is not None:
        values["embedding_model"] = embedding_model
        values["embedding_dimensions"] = str(EMBEDDING_DIMENSIONS)
    for key, value in values.items():
        set_meta(connection, key, value)


def insert_lexical_document(
    connection: sqlite3.Connection, document: FactDocument
) -> None:
    title = document.title or _metadata_title(document.metadata) or document.document_id
    document_metadata = _flatten_metadata(document.metadata)
    for chunk in sorted(document.chunks, key=lambda item: item.ordinal):
        chunk_metadata = _flatten_metadata(chunk.metadata)
        searchable_metadata = "\n".join(
            value for value in (document_metadata, chunk_metadata) if value
        )
        connection.execute(
            """
            INSERT INTO chunks(
                chunk_id, document_id, ordinal, title, headings, metadata, body, locator_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.ordinal,
                title,
                "\n".join(chunk.headings),
                searchable_metadata,
                chunk.raw_text,
                chunk.locator.model_dump_json(),
            ),
        )


def insert_semantic_document(
    connection: sqlite3.Connection, document: FactDocument
) -> list[int]:
    return insert_semantic_segments(
        connection,
        [
            (chunk.locator, 0)
            for chunk in sorted(document.chunks, key=lambda item: item.ordinal)
        ],
    )


def insert_semantic_segments(
    connection: sqlite3.Connection,
    segments: Sequence[tuple[DocumentChunkLocator, int]],
) -> list[int]:
    rowids: list[int] = []
    for locator, embedding_segment_ordinal in segments:
        if (
            isinstance(embedding_segment_ordinal, bool)
            or not isinstance(embedding_segment_ordinal, int)
            or embedding_segment_ordinal < 0
        ):
            raise ValueError("embedding_segment_ordinal must be a non-negative integer")
        cursor = connection.execute(
            """
            INSERT INTO chunks(
                chunk_id, document_id, ordinal,
                embedding_segment_ordinal, locator_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                locator.chunk_id,
                locator.document_id,
                locator.ordinal,
                embedding_segment_ordinal,
                locator.model_dump_json(),
            ),
        )
        rowids.append(int(cursor.lastrowid))
    return rowids


def insert_vectors(
    connection: sqlite3.Connection,
    rowids: Sequence[int],
    embeddings: Sequence[Sequence[float]],
) -> None:
    if len(rowids) != len(embeddings):
        raise ValueError("rowids and embeddings must have equal lengths")
    validated = validate_embeddings(embeddings, expected_count=len(rowids))
    connection.executemany(
        "INSERT INTO chunk_vectors(chunk_rowid, embedding) VALUES (?, ?)",
        (
            (rowid, serialize_float32(list(embedding)))
            for rowid, embedding in zip(rowids, validated, strict=True)
        ),
    )


def optimize_and_check_fts(connection: sqlite3.Connection) -> None:
    connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
    connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('integrity-check')")


def keyword_search(
    connection: sqlite3.Connection,
    rewrite: QueryRewrite,
    *,
    limit: int = MAX_CANDIDATES,
) -> tuple[RetrievalHit, ...]:
    _validate_limit(limit)
    status = get_index_status(connection)
    if status.index_kind != "lexical" or status.status != "ready":
        raise IndexNotReadyError(
            "LEXICAL_INDEX_NOT_READY", "lexical index is not ready"
        )
    match = compile_fts_match(rewrite)
    if match is None:
        return ()
    rows = connection.execute(
        """
        SELECT chunks.locator_json AS locator_json,
               bm25(chunks_fts, 5.0, 3.0, 2.0, 1.0) AS score,
               chunks.chunk_id AS chunk_id
        FROM chunks_fts
        JOIN chunks ON chunks.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY score ASC, chunk_id ASC
        LIMIT ?
        """,
        (match, limit),
    ).fetchall()
    return tuple(
        RetrievalHit(
            provider="keyword",
            rank=rank,
            native_score=float(row["score"]),
            locator=DocumentChunkLocator.model_validate_json(row["locator_json"]),
        )
        for rank, row in enumerate(rows, start=1)
    )


def vector_search(
    connection: sqlite3.Connection,
    query_embedding: Sequence[float],
    *,
    limit: int = MAX_CANDIDATES,
) -> tuple[RetrievalHit, ...]:
    _validate_limit(limit)
    status = get_index_status(connection)
    if status.index_kind != "semantic" or status.status not in {"ready", "test_only"}:
        raise IndexNotReadyError(
            "SEMANTIC_INDEX_NOT_READY", "semantic index is not ready"
        )
    values = validate_embeddings([query_embedding], expected_count=1)[0]
    total_segments = int(
        connection.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0]
    )
    if total_segments == 0:
        return ()
    query_blob = serialize_float32(list(values))
    candidate_count = min(total_segments, max(limit, limit * 2))
    unique_rows: list[sqlite3.Row] = []
    while True:
        rows = connection.execute(
            """
            WITH nearest AS (
                SELECT chunk_rowid, distance
                FROM chunk_vectors
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT chunks.locator_json AS locator_json,
                   nearest.distance AS distance,
                   chunks.chunk_id AS chunk_id,
                   chunks.embedding_segment_ordinal AS embedding_segment_ordinal
            FROM nearest
            JOIN chunks ON chunks.id = nearest.chunk_rowid
            ORDER BY distance ASC, chunk_id ASC, embedding_segment_ordinal ASC
            """,
            (query_blob, candidate_count),
        ).fetchall()
        seen_chunks: set[str] = set()
        unique_rows = []
        for row in rows:
            chunk_id = str(row["chunk_id"])
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            unique_rows.append(row)
            if len(unique_rows) == limit:
                break
        if len(unique_rows) == limit or candidate_count == total_segments:
            break
        candidate_count = min(total_segments, candidate_count * 2)
    return tuple(
        RetrievalHit(
            provider="vector",
            rank=rank,
            native_score=float(row["distance"]),
            locator=DocumentChunkLocator.model_validate_json(row["locator_json"]),
            retrieval_segment_ordinal=int(row["embedding_segment_ordinal"]),
        )
        for rank, row in enumerate(unique_rows, start=1)
    )


def _metadata_title(metadata: dict[str, JsonValue]) -> str | None:
    value = metadata.get("title")
    return value if isinstance(value, str) and value.strip() else None


def _flatten_metadata(metadata: dict[str, JsonValue]) -> str:
    values: list[str] = []

    def visit(value: JsonValue) -> None:
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, (str, int, float)):
            values.append(str(value))
            return
        if isinstance(value, list):
            for child in value:
                visit(child)
            return
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key])

    visit(metadata)
    return "\n".join(values)


def _validate_limit(limit: int) -> None:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_CANDIDATES
    ):
        raise ValueError(f"limit must be between 1 and {MAX_CANDIDATES}")
