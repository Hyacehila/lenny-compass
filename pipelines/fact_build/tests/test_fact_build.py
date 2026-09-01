from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ai_pm_contracts import FactDocument
from docling_core.types.doc import DoclingDocument
from fact_build.builder import (
    FactBuildError,
    _assert_chunk_integrity,
    _document_paths,
    _frontmatter,
    _markdown_images,
    _publish_fact_generation,
)

REQUIRED_PILOT = {
    "02-newsletters/get-perplexity-pro-free-for-1-year.md",
    "02-newsletters/how-to-kickstart-and-scale-a-consumer-business.md",
    "02-newsletters/how-to-measure-cohort-retention.md",
    "02-newsletters/how-to-run-seo-experiments.md",
    "03-podcasts/eric-ries.md",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_configuration_locks_pipeline_and_exact_pilot(fact_config) -> None:
    assert set(fact_config.pilot_documents) == REQUIRED_PILOT
    assert fact_config.max_tokens == 1500
    assert fact_config.expected_document_count == 638
    assert fact_config.config_version == "fact-build-v2"
    assert fact_config.data["output"]["artifact_version"] == "v2"
    assert fact_config.data["docling"]["pipeline"] == "SimplePipeline"
    assert fact_config.data["docling"]["fetch_remote_images"] is False
    assert fact_config.data["chunking"] == {
        "chunker": "HybridChunker",
        "tokenizer_model": "sentence-transformers/all-MiniLM-L6-v2",
        "tokenizer_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "max_tokens": 1500,
        "merge_peers": True,
    }


def test_generation_paths_preserve_v2_layout_beneath_build_id(fact_config) -> None:
    document_id = "02-newsletters/example.md"
    _docling, _chunks, docling_relative, chunks_relative = _document_paths(
        fact_config,
        document_id,
        artifact_prefix=Path("generations") / "fact-build-v2-example",
    )
    assert docling_relative == (
        "generations/fact-build-v2-example/documents/v2/"
        "02-newsletters/example.docling.json"
    )
    assert chunks_relative == (
        "generations/fact-build-v2-example/chunks/v2/"
        "02-newsletters/example.chunks.json"
    )


def test_conversion_can_bind_all_fact_locators_to_one_generation(fact_builder) -> None:
    prefix = Path("generations") / "fact-build-v2-example"
    built = fact_builder.convert_one(
        "02-newsletters/get-perplexity-pro-free-for-1-year.md",
        artifact_prefix=prefix,
    )
    document = FactDocument.model_validate_json(built.fact_bytes)
    expected_root = "generations/fact-build-v2-example/"

    assert document.docling_path.startswith(expected_root)
    assert document.fact_document_path.startswith(expected_root)
    assert built.manifest_entry.docling_path == document.docling_path
    assert built.manifest_entry.fact_document_path == document.fact_document_path
    assert built.manifest_entry.fact_document_sha256 == sha256(built.fact_bytes)
    assert all(
        chunk.locator.docling_path == document.docling_path
        and chunk.locator.fact_document_path == document.fact_document_path
        for chunk in document.chunks
    )


def test_generation_publish_rolls_back_if_manifest_pointer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts_root = tmp_path / "facts"
    staging_root = facts_root / ".staging"
    build_id = "fact-build-v2-candidate"
    staged_generation = staging_root / "generations" / build_id
    staged_generation.mkdir(parents=True)
    (staged_generation / "complete.txt").write_text("candidate", encoding="utf-8")
    manifest_path = facts_root / "manifests" / "build.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(b"previous-manifest")

    def fail_pointer(_path: Path, _data: bytes) -> None:
        raise OSError("simulated pointer failure")

    monkeypatch.setattr("fact_build.builder._safe_write", fail_pointer)
    with pytest.raises(OSError, match="simulated pointer failure"):
        _publish_fact_generation(
            facts_root=facts_root,
            staging_root=staging_root,
            build_id=build_id,
            manifest_path=manifest_path,
            manifest_bytes=b"candidate-manifest",
        )

    assert manifest_path.read_bytes() == b"previous-manifest"
    assert not (facts_root / "generations" / build_id).exists()


def test_provenance_has_exact_corpus_and_byte_hashes(fact_config) -> None:
    manifest_path = fact_config.input_path("enhanced_provenance")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["documents"]) == 638
    assert len({item["document_id"] for item in manifest["documents"]}) == 638
    for item in manifest["documents"]:
        raw = fact_config.repo_path(item["raw"]["path"]).read_bytes()
        enhanced = fact_config.repo_path(item["enhanced"]["path"]).read_bytes()
        assert sha256(raw) == item["raw"]["sha256"]
        assert sha256(enhanced) == item["enhanced"]["sha256"]
        assert _frontmatter(raw, item["raw"]["path"]) == item["metadata"]


def test_small_document_contract_metadata_and_locators(fact_builder) -> None:
    document_id = "02-newsletters/get-perplexity-pro-free-for-1-year.md"
    built = fact_builder.convert_one(document_id)
    validated = FactDocument.model_validate_json(built.fact_bytes)
    docling = DoclingDocument.model_validate_json(built.docling_bytes)

    assert validated == built.document
    assert docling.model_dump(mode="json")
    assert validated.document_id == document_id
    assert validated.raw_path == f"corpus/raw/{document_id}"
    assert validated.enhanced_path == f"corpus/enhanced/{document_id}"
    assert validated.raw_sha256 == fact_builder.records[document_id].raw_sha256
    assert validated.enhanced_sha256 == fact_builder.records[document_id].enhanced_sha256
    assert validated.metadata["title"] == "Get Perplexity Pro free for 1 year"
    assert validated.metadata["type"] == "newsletter"
    assert validated.docling_path == (
        "documents/v2/02-newsletters/get-perplexity-pro-free-for-1-year.docling.json"
    )
    assert validated.fact_document_path == (
        "chunks/v2/02-newsletters/get-perplexity-pro-free-for-1-year.chunks.json"
    )
    assert validated.chunks[0].locator.fact_document_path == (
        "chunks/v2/02-newsletters/get-perplexity-pro-free-for-1-year.chunks.json"
    )
    assert validated.chunks[0].locator.raw_sha256 == fact_builder.records[document_id].raw_sha256
    assert validated.chunks[0].locator.raw_path == validated.raw_path
    assert validated.chunks[0].locator.enhanced_path == validated.enhanced_path
    assert validated.chunks[0].locator.segment_ordinal == 0
    assert built.manifest_entry.fact_document_sha256 == sha256(built.fact_bytes)
    assert built.manifest_entry.fact_document_sha256 != sha256(built.fact_bytes + b" ")
    assert built.manifest_entry.docling_sha256 == sha256(built.docling_bytes)
    assert built.metrics["checks"]["short_without_section_heading"] is True
    assert all(chunk.token_count <= 1500 for chunk in validated.chunks)


def test_markdown_image_parser_skips_code_and_preserves_caption() -> None:
    markdown = """![real caption](https://example.test/a.png)\n\n```md\n![not real](x.png)\n```\n"""
    assert _markdown_images(markdown) == (
        # A small value-object comparison also guards ordering.
        type(_markdown_images(markdown)[0])(
            url="https://example.test/a.png",
            caption="real caption",
        ),
    )


def test_html_edge_preserves_image_urls_and_captions(fact_builder) -> None:
    document_id = "02-newsletters/how-to-run-seo-experiments.md"
    source = fact_builder.config.repo_path(fact_builder.records[document_id].enhanced_path)
    source_images = _markdown_images(source.read_text(encoding="utf-8"))
    built = fact_builder.convert_one(document_id)
    document = FactDocument.model_validate_json(built.fact_bytes)
    docling = DoclingDocument.model_validate_json(built.docling_bytes)

    chunk_images = [image for chunk in document.chunks for image in chunk.images]
    assert [image.url for image in chunk_images] == [image.url for image in source_images]
    assert [image.caption for image in chunk_images] == [image.caption for image in source_images]
    assert [str(picture.image.uri) for picture in docling.pictures] == [
        image.url for image in source_images
    ]
    assert built.metrics["checks"]["native_html_preserved"] is True


def test_existing_docling_caption_is_not_duplicated(fact_builder) -> None:
    document_id = "02-newsletters/autonomy-vs-direction---issue-35.md"
    built = fact_builder.convert_one(document_id)
    docling = DoclingDocument.model_validate_json(built.docling_bytes)
    assert len(docling.pictures) == 1
    assert [ref.resolve(docling).text for ref in docling.pictures[0].captions] == [
        "Image for post"
    ]


def test_chunk_content_tamper_is_rejected_by_canonical_hash(fact_builder) -> None:
    built = fact_builder.convert_one(
        "02-newsletters/get-perplexity-pro-free-for-1-year.md"
    )
    payload = json.loads(built.fact_bytes)
    payload["chunks"][0]["raw_text"] += " tampered"
    tampered = FactDocument.model_validate(payload)
    with pytest.raises(FactBuildError, match="canonical chunk hash mismatch"):
        _assert_chunk_integrity(tampered.chunks[0])


def test_code_and_podcast_structure_pilot_cases(fact_builder) -> None:
    code = fact_builder.convert_one("02-newsletters/how-to-measure-cohort-retention.md")
    podcast = fact_builder.convert_one("03-podcasts/eric-ries.md")
    assert code.metrics["checks"]["code_items_preserved"] is True
    assert podcast.metrics["checks"]["podcast_heading_hierarchy"] is True
    assert max(chunk.token_count for chunk in podcast.document.chunks) <= 1500
