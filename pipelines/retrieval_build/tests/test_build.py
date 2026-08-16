from __future__ import annotations

from pathlib import Path

import pytest
from ai_pm_contracts import DocumentChunkLocator
from conftest import ThresholdEmbeddingProvider, make_document, write_manifest
from retrieval_build import FactStore
from retrieval_build.build import (
    build_retrieval_bundle,
    load_and_validate_bundle_manifest,
)
from retrieval_build.database import connect_lexical, connect_semantic, get_index_status
from retrieval_build.embeddings import MockEmbeddingProvider
from retrieval_build.errors import IndexBuildError


def _outputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "retrieval"
    return (
        root / "lexical" / "fts.sqlite3",
        root / "semantic" / "vec.sqlite3",
        root / "manifests" / "build.json",
    )


def test_lexical_build_publishes_without_a_fake_semantic_database(
    tmp_path: Path,
) -> None:
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    document = make_document(
        "build-doc", ["product strategy", "growth loops"], title="Build"
    )
    facts_manifest = write_manifest(facts_root, [document])
    lexical, semantic, bundle_manifest = _outputs(tmp_path)

    manifest = build_retrieval_bundle(
        facts_manifest_path=facts_manifest,
        lexical_output_path=lexical,
        semantic_output_path=semantic,
        manifest_output_path=bundle_manifest,
        semantic_reason="ZHIPU_API_KEY is not set",
    )

    assert facts_manifest == facts_root / "manifests" / "build.json"
    assert (facts_root / "chunks" / "v1" / "build-doc.json").exists()
    assert manifest.lexical_status == "ready"
    assert manifest.facts_manifest_file == "facts/manifests/build.json"
    assert manifest.semantic_status == "not_ready"
    assert manifest.semantic_reason == "ZHIPU_API_KEY is not set"
    assert lexical.exists()
    assert not semantic.exists()
    assert bundle_manifest.exists()
    assert manifest.semantic_database_file is None
    assert load_and_validate_bundle_manifest(bundle_manifest) == manifest
    with connect_lexical(lexical, query_only=True) as connection:
        status = get_index_status(connection)
        assert status.index_kind == "lexical"
        assert status.status == "ready"
        assert connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT count(*) FROM sqlite_master WHERE name = 'chunk_vectors'"
            ).fetchone()[0]
            == 0
        )


def test_mock_build_creates_independent_test_only_vector_database(
    tmp_path: Path,
) -> None:
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    facts_manifest = write_manifest(
        facts_root,
        [make_document("mock-doc", ["one", "two"], title="Mock")],
    )
    lexical, semantic, bundle_manifest = _outputs(tmp_path)
    manifest = build_retrieval_bundle(
        facts_manifest_path=facts_manifest,
        lexical_output_path=lexical,
        semantic_output_path=semantic,
        manifest_output_path=bundle_manifest,
        embedder=MockEmbeddingProvider(),
        semantic_reason="explicit test-only mock embeddings",
    )
    assert manifest.semantic_status == "test_only"
    assert lexical.exists() and semantic.exists()
    assert lexical != semantic
    with connect_lexical(lexical, query_only=True) as lexical_connection:
        lexical_status = get_index_status(lexical_connection)
    with connect_semantic(semantic, query_only=True) as semantic_connection:
        semantic_status = get_index_status(semantic_connection)
        assert (
            semantic_connection.execute(
                "SELECT count(*) FROM chunk_vectors"
            ).fetchone()[0]
            == 2
        )
    assert lexical_status.build_id == semantic_status.build_id == manifest.build_id
    assert (
        lexical_status.facts_manifest_sha256
        == semantic_status.facts_manifest_sha256
        == manifest.facts_manifest_sha256
    )
    assert load_and_validate_bundle_manifest(bundle_manifest) == manifest


def test_bundle_manifest_detects_lexical_database_tampering(tmp_path: Path) -> None:
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    facts_manifest = write_manifest(facts_root, [make_document("tamper", ["one"])])
    lexical, semantic, bundle_manifest = _outputs(tmp_path)
    build_retrieval_bundle(
        facts_manifest_path=facts_manifest,
        lexical_output_path=lexical,
        semantic_output_path=semantic,
        manifest_output_path=bundle_manifest,
    )
    with lexical.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(IndexBuildError) as caught:
        load_and_validate_bundle_manifest(bundle_manifest)
    assert caught.value.code == "STALE_BUILD_MANIFEST"


def test_build_splits_only_overlong_embedding_input_and_keeps_parent_locator(
    tmp_path: Path,
) -> None:
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    long_text = (
        "Define activation around retained value, not a superficial click.\n\n"
        "Validate the event against cohorts and qualitative user research.\n\n"
        "Revisit the definition when the product and target audience change."
    )
    document = make_document("overlong", [long_text], title="Activation")
    facts_manifest = write_manifest(facts_root, [document])
    lexical, semantic, bundle_manifest = _outputs(tmp_path)

    manifest = build_retrieval_bundle(
        facts_manifest_path=facts_manifest,
        lexical_output_path=lexical,
        semantic_output_path=semantic,
        manifest_output_path=bundle_manifest,
        embedder=ThresholdEmbeddingProvider(maximum_characters=75),
        semantic_reason="test-only threshold provider",
    )

    assert manifest.semantic_status == "test_only"
    canonical = document.chunks[0].locator
    with connect_semantic(semantic, query_only=True) as connection:
        rows = connection.execute(
            """
            SELECT embedding_segment_ordinal, locator_json
            FROM chunks
            ORDER BY embedding_segment_ordinal
            """
        ).fetchall()
        assert len(rows) > 1
        assert [row["embedding_segment_ordinal"] for row in rows] == list(
            range(len(rows))
        )
        locators = [
            DocumentChunkLocator.model_validate_json(row["locator_json"])
            for row in rows
        ]
        assert locators == [canonical] * len(rows)

    store = FactStore(facts_root)
    assert all(
        store.read_source_window(locator, window=0)[0].locator == canonical
        for locator in locators
    )


def test_not_ready_rebuild_removes_stale_semantic_database(tmp_path: Path) -> None:
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    facts_manifest = write_manifest(facts_root, [make_document("stale-vec", ["one"])])
    lexical, semantic, bundle_manifest = _outputs(tmp_path)
    build_retrieval_bundle(
        facts_manifest_path=facts_manifest,
        lexical_output_path=lexical,
        semantic_output_path=semantic,
        manifest_output_path=bundle_manifest,
        embedder=MockEmbeddingProvider(),
    )
    assert semantic.exists()

    manifest = build_retrieval_bundle(
        facts_manifest_path=facts_manifest,
        lexical_output_path=lexical,
        semantic_output_path=semantic,
        manifest_output_path=bundle_manifest,
        semantic_reason="semantic disabled for this generation",
    )
    assert manifest.semantic_status == "not_ready"
    assert not semantic.exists()
    assert load_and_validate_bundle_manifest(bundle_manifest) == manifest


def test_build_rejects_manifest_entry_artifact_hash_mismatch(tmp_path: Path) -> None:
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    document = make_document("tampered-source", ["one"])
    facts_manifest = write_manifest(facts_root, [document])
    (tmp_path / document.enhanced_path).write_bytes(b"tampered")
    lexical, semantic, bundle_manifest = _outputs(tmp_path)

    with pytest.raises(IndexBuildError) as caught:
        build_retrieval_bundle(
            facts_manifest_path=facts_manifest,
            lexical_output_path=lexical,
            semantic_output_path=semantic,
            manifest_output_path=bundle_manifest,
        )
    assert caught.value.code == "FACT_ARTIFACT_HASH_MISMATCH"
    assert not lexical.exists()
    assert not bundle_manifest.exists()
