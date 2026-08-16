from __future__ import annotations

from pathlib import Path

from ai_pm_contracts import QueryRewrite
from conftest import make_document, sha
from retrieval_build import FactStore
from retrieval_build.database import (
    check_fts5_available,
    connect_lexical,
    connect_semantic,
    create_lexical_schema,
    create_semantic_schema,
    get_index_status,
    initialize_index_meta,
    insert_lexical_document,
    insert_semantic_document,
    insert_semantic_segments,
    insert_vectors,
    keyword_search,
    set_meta,
    vector_search,
)
from retrieval_build.embeddings import MockEmbeddingProvider


def test_fts5_bm25_weights_title_and_preserves_proper_names(tmp_path: Path) -> None:
    database = tmp_path / "fts.sqlite3"
    with connect_lexical(database) as connection:
        create_lexical_schema(connection)
        initialize_index_meta(
            connection,
            index_kind="lexical",
            build_id="build-1",
            facts_manifest_sha256=sha("facts"),
        )
        title_document = make_document(
            "title-hit",
            ["A short discussion of product focus."],
            title="Steve Jobs interview",
        )
        body_document = make_document(
            "body-hit",
            ["Steve Jobs Steve Jobs discussed hiring and management."],
            title="Hiring advice",
        )
        insert_lexical_document(connection, title_document)
        insert_lexical_document(connection, body_document)
        set_meta(connection, "status", "ready")
        connection.commit()

        rewrite = QueryRewrite(
            original_query="Steve Jobs",
            semantic_query_en="Steve Jobs",
            lexical_phrases_en=("Steve Jobs",),
        )
        hits = keyword_search(connection, rewrite)
        assert hits[0].locator.document_id == "title-hit"
        assert "text" not in hits[0].model_dump()

        singular = QueryRewrite(
            original_query="job",
            semantic_query_en="job",
            lexical_terms_en=("job",),
        )
        assert keyword_search(connection, singular) == ()


def test_sqlite_vec_cosine_known_nearest_neighbor(tmp_path: Path) -> None:
    database = tmp_path / "vec.sqlite3"
    embedder = MockEmbeddingProvider()
    document = make_document(
        "vector-doc",
        ["apple orchard fruit", "banana tropical fruit", "leadership management"],
        title="Vector test",
    )
    with connect_semantic(database) as connection:
        create_semantic_schema(connection)
        initialize_index_meta(
            connection,
            index_kind="semantic",
            build_id="build-1",
            facts_manifest_sha256=sha("facts"),
            embedding_model=embedder.model_name,
        )
        rowids = insert_semantic_document(connection, document)
        insert_vectors(
            connection,
            rowids,
            embedder.embed_all(
                [chunk.contextualized_text for chunk in document.chunks]
            ),
        )
        set_meta(connection, "status", "test_only")
        connection.commit()

        query = embedder.embed([document.chunks[0].contextualized_text])[0]
        hits = vector_search(connection, query)
        assert hits[0].locator.chunk_id == document.chunks[0].chunk_id
        assert hits[0].native_score <= 1e-6
        assert "text" not in hits[0].model_dump()


def test_each_database_has_independent_validated_status(tmp_path: Path) -> None:
    lexical = tmp_path / "lexical.sqlite3"
    semantic = tmp_path / "semantic.sqlite3"
    facts_hash = sha("facts")
    with connect_lexical(lexical) as connection:
        check_fts5_available(connection)
        create_lexical_schema(connection)
        initialize_index_meta(
            connection,
            index_kind="lexical",
            build_id="build-1",
            facts_manifest_sha256=facts_hash,
            status="ready",
        )
        status = get_index_status(connection)
        assert status.index_kind == "lexical"
        assert status.status == "ready"
        assert status.embedding_dimensions is None

    with connect_semantic(semantic) as connection:
        create_semantic_schema(connection)
        initialize_index_meta(
            connection,
            index_kind="semantic",
            build_id="build-1",
            facts_manifest_sha256=facts_hash,
            status="test_only",
            embedding_model="mock-hash-embedding",
        )
        status = get_index_status(connection)
        assert status.index_kind == "semantic"
        assert status.status == "test_only"
        assert status.embedding_dimensions == 2048


def test_vector_segments_deduplicate_parent_and_fact_store_reads_canonical_chunk(
    fact_artifacts,
    tmp_path: Path,
) -> None:
    facts_root, document = fact_artifacts
    database = tmp_path / "segmented.sqlite3"
    first = document.chunks[0].locator
    second = document.chunks[1].locator
    exact = [0.0] * 2048
    exact[0] = 1.0
    nearly_exact = [0.0] * 2048
    nearly_exact[0] = 0.99
    nearly_exact[1] = 0.01
    other = [0.0] * 2048
    other[1] = 1.0

    with connect_semantic(database) as connection:
        create_semantic_schema(connection)
        initialize_index_meta(
            connection,
            index_kind="semantic",
            build_id="build-segments",
            facts_manifest_sha256=sha("facts"),
            status="test_only",
            embedding_model="mock-hash-embedding",
        )
        rowids = insert_semantic_segments(
            connection,
            [(first, 0), (first, 1), (second, 0)],
        )
        insert_vectors(connection, rowids, [exact, nearly_exact, other])
        connection.commit()

        hits = vector_search(connection, exact, limit=2)
        assert [hit.locator.chunk_id for hit in hits] == [
            first.chunk_id,
            second.chunk_id,
        ]
        assert hits[0].retrieval_segment_ordinal == 0
        assert hits[0].locator == first
        assert get_index_status(connection).status == "test_only"

    evidence = FactStore(facts_root).read_source_window(hits[0].locator, window=0)
    assert len(evidence) == 1
    assert evidence[0].locator == first
