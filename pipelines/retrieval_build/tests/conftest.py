from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from ai_pm_contracts import (
    DocumentChunkLocator,
    DocumentSegment,
    FactBuildManifest,
    FactBuildManifestEntry,
    FactChunkRecord,
    FactDocument,
    canonical_chunk_sha256,
)
from retrieval_build.embeddings import (
    EmbeddingInputTooLongError,
    MockEmbeddingProvider,
)


class ThresholdEmbeddingProvider(MockEmbeddingProvider):
    """Test provider that reports only deterministic per-input length errors."""

    model_name = "threshold-test-embedding"

    def __init__(self, maximum_characters: int):
        self.maximum_characters = maximum_characters
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts):
        self.calls.append(tuple(texts))
        if any(len(text) > self.maximum_characters for text in texts):
            raise EmbeddingInputTooLongError("test input is too long")
        return super().embed(texts)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _raw_text(document_id: str) -> str:
    return f"raw source for {document_id}\n"


def _enhanced_text(document_id: str) -> str:
    return f"enhanced source for {document_id}\n"


def _docling_bytes(document_id: str) -> bytes:
    return f'{{"document_id":"{document_id}"}}\n'.encode()


def make_document(
    document_id: str,
    texts: list[str],
    *,
    title: str | None = None,
    fact_document_path: str | None = None,
    metadata: dict | None = None,
) -> FactDocument:
    artifact_path = fact_document_path or f"chunks/v1/{document_id}.json"
    raw_path = f"corpus/raw/{document_id}.md"
    raw_hash = sha(_raw_text(document_id))
    enhanced_path = f"corpus/enhanced/{document_id}.md"
    enhanced_hash = sha(_enhanced_text(document_id))
    docling_path = f"documents/v1/{document_id}.json"
    docling_hash = hashlib.sha256(_docling_bytes(document_id)).hexdigest()
    chunks: list[FactChunkRecord] = []
    for ordinal, text in enumerate(texts):
        self_ref = f"#/texts/{ordinal}"
        chunk_id = f"{document_id}-chunk-{ordinal}"
        headings = (title,) if title else ()
        contextualized_text = f"{title or document_id}\n{text}"
        chunk_hash = canonical_chunk_sha256(
            raw_text=text,
            contextualized_text=contextualized_text,
            headings=headings,
            images=(),
            self_refs=(self_ref,),
        )
        locator = DocumentChunkLocator(
            document_id=document_id,
            chunk_id=chunk_id,
            ordinal=ordinal,
            segment_ordinal=0,
            fact_document_path=artifact_path,
            raw_path=raw_path,
            raw_sha256=raw_hash,
            enhanced_path=enhanced_path,
            enhanced_sha256=enhanced_hash,
            docling_path=docling_path,
            docling_sha256=docling_hash,
            chunk_sha256=chunk_hash,
            segment=DocumentSegment(
                first_self_ref=self_ref,
                last_self_ref=self_ref,
                item_count=1,
            ),
            self_refs=(self_ref,),
            headings=headings,
        )
        chunks.append(
            FactChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                ordinal=ordinal,
                raw_text=text,
                contextualized_text=contextualized_text,
                token_count=max(1, len(text.split())),
                headings=headings,
                metadata={},
                doc_item_refs=(self_ref,),
                locator=locator,
            )
        )
    return FactDocument(
        document_id=document_id,
        raw_path=raw_path,
        raw_sha256=raw_hash,
        enhanced_path=enhanced_path,
        enhanced_sha256=enhanced_hash,
        docling_path=docling_path,
        docling_sha256=docling_hash,
        fact_document_path=artifact_path,
        title=title,
        metadata=metadata or {},
        chunks=tuple(chunks),
    )


@pytest.fixture
def fact_artifacts(tmp_path: Path) -> tuple[Path, FactDocument]:
    root = tmp_path / "facts"
    root.mkdir()
    document = make_document(
        "alpha",
        ["zero block", "one target block", "two block", "three block"],
        title="Alpha document",
    )
    write_manifest(root, [document])
    return root, document


def write_manifest(root: Path, documents: list[FactDocument]) -> Path:
    entries: list[FactBuildManifestEntry] = []
    for document in documents:
        relative_path = document.fact_document_path
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fact_bytes = document.model_dump_json(indent=2).encode("utf-8")
        path.write_bytes(fact_bytes)
        raw_path = root.parent / document.raw_path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(_raw_text(document.document_id).encode("utf-8"))
        enhanced_path = root.parent / document.enhanced_path
        enhanced_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced_path.write_bytes(_enhanced_text(document.document_id).encode("utf-8"))
        docling_path = root / document.docling_path
        docling_path.parent.mkdir(parents=True, exist_ok=True)
        docling_path.write_bytes(_docling_bytes(document.document_id))
        entries.append(
            FactBuildManifestEntry(
                document_id=document.document_id,
                raw_path=document.raw_path,
                raw_sha256=document.raw_sha256,
                enhanced_path=document.enhanced_path,
                enhanced_sha256=document.enhanced_sha256,
                fact_document_path=relative_path,
                fact_document_sha256=hashlib.sha256(fact_bytes).hexdigest(),
                docling_path=document.docling_path,
                docling_sha256=document.docling_sha256,
                chunk_count=len(document.chunks),
            )
        )
    manifest = FactBuildManifest(
        build_id="facts-test-build",
        created_at=datetime.now(UTC),
        document_count=len(entries),
        chunk_count=sum(entry.chunk_count for entry in entries),
        documents=tuple(entries),
    )
    path = root / "manifests" / "build.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path
