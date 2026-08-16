from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from ._base import ContractModel

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, Field(min_length=1)]


class ImageReference(ContractModel):
    """Stable reference to an image retained by a Docling fact artifact."""

    image_id: NonEmpty
    self_ref: str | None = None
    path: str | None = None
    url: str | None = None
    sha256: Sha256 | None = None
    caption: str | None = None


class ImageObservation(ContractModel):
    """Typed result of an optional vision inspection for cited evidence."""

    image_id: NonEmpty
    url: str | None = None
    caption: str | None = None
    observation: str | None = None
    inspected: bool = False
    error: str | None = None


class DocumentSegment(ContractModel):
    """Inclusive Docling item span represented by a chunk."""

    first_self_ref: NonEmpty
    last_self_ref: NonEmpty
    item_count: Annotated[int, Field(ge=1)]


class DocumentChunkLocator(ContractModel):
    """Versioned, resolvable pointer from a retrieval hit to source facts.

    Paths are repository-relative. They are treated as untrusted input and are
    resolved beneath a configured facts root by ``FactStore``.
    """

    kind: Literal["document_chunk"] = "document_chunk"
    config_version: NonEmpty = "1"
    document_id: NonEmpty
    chunk_id: NonEmpty
    ordinal: Annotated[int, Field(ge=0)]
    segment_ordinal: Annotated[int, Field(ge=0)]
    fact_document_path: NonEmpty
    raw_path: NonEmpty
    raw_sha256: Sha256
    enhanced_path: NonEmpty
    enhanced_sha256: Sha256
    docling_path: NonEmpty
    docling_sha256: Sha256
    chunk_sha256: Sha256
    segment: DocumentSegment
    self_refs: tuple[str, ...] = ()
    headings: tuple[str, ...] = ()
    images: tuple[ImageReference, ...] = ()

    @model_validator(mode="after")
    def validate_segment_refs(self) -> DocumentChunkLocator:
        if self.self_refs:
            if self.self_refs[0] != self.segment.first_self_ref:
                raise ValueError("segment.first_self_ref must match self_refs[0]")
            if self.self_refs[-1] != self.segment.last_self_ref:
                raise ValueError("segment.last_self_ref must match self_refs[-1]")
            if len(self.self_refs) != self.segment.item_count:
                raise ValueError("segment.item_count must match len(self_refs)")
        return self


class LlmWikiPageLocator(ContractModel):
    """Reserved pointer into a future compiled LLM Wiki knowledge layer."""

    kind: Literal["llm_wiki_page"] = "llm_wiki_page"
    project_id: NonEmpty
    path: NonEmpty
    revision: NonEmpty | None = None
    fact_id: str | None = None
    provider_uri: str | None = None


EvidenceLocator = Annotated[
    DocumentChunkLocator | LlmWikiPageLocator,
    Field(discriminator="kind"),
]


class Evidence(ContractModel):
    """Fact-layer content that the answer model is permitted to cite."""

    evidence_id: NonEmpty
    evidence_kind: Literal["source_chunk", "wiki_fact"]
    locator: EvidenceLocator
    text: NonEmpty
    title: str | None = None
    headings: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    doc_item_refs: tuple[str, ...] = ()
    image_observations: tuple[ImageObservation, ...] = ()

    @model_validator(mode="after")
    def validate_locator_kind(self) -> Evidence:
        if self.evidence_kind == "source_chunk" and not isinstance(
            self.locator, DocumentChunkLocator
        ):
            raise ValueError("source_chunk evidence requires DocumentChunkLocator")
        if self.evidence_kind == "wiki_fact" and not isinstance(
            self.locator, LlmWikiPageLocator
        ):
            raise ValueError("wiki_fact evidence requires LlmWikiPageLocator")
        return self


class FactChunkRecord(ContractModel):
    """One structure-preserving HybridChunker output."""

    chunk_id: NonEmpty
    document_id: NonEmpty
    ordinal: Annotated[int, Field(ge=0)]
    raw_text: NonEmpty
    contextualized_text: NonEmpty
    token_count: Annotated[int, Field(ge=1)]
    headings: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    doc_item_refs: tuple[str, ...] = ()
    images: tuple[ImageReference, ...] = ()
    locator: DocumentChunkLocator

    @model_validator(mode="after")
    def validate_locator_alignment(self) -> FactChunkRecord:
        if self.locator.chunk_id != self.chunk_id:
            raise ValueError("locator.chunk_id does not match chunk_id")
        if self.locator.document_id != self.document_id:
            raise ValueError("locator.document_id does not match document_id")
        if self.locator.ordinal != self.ordinal:
            raise ValueError("locator.ordinal does not match ordinal")
        if self.locator.self_refs != self.doc_item_refs:
            raise ValueError("locator.self_refs does not match doc_item_refs")
        return self


class FactDocument(ContractModel):
    """Self-contained, versioned fact artifact for a single source document."""

    schema_version: Literal["1"] = "1"
    config_version: NonEmpty = "1"
    document_id: NonEmpty
    raw_path: NonEmpty
    raw_sha256: Sha256
    enhanced_path: NonEmpty
    enhanced_sha256: Sha256
    docling_path: NonEmpty
    docling_sha256: Sha256
    fact_document_path: NonEmpty
    title: str | None = None
    document_type: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    chunks: tuple[FactChunkRecord, ...]

    @model_validator(mode="after")
    def validate_chunks(self) -> FactDocument:
        ordinals: set[int] = set()
        chunk_ids: set[str] = set()
        for chunk in self.chunks:
            if chunk.document_id != self.document_id:
                raise ValueError("chunk document_id does not match fact document")
            identity_pairs = (
                (chunk.locator.raw_path, self.raw_path, "raw path"),
                (chunk.locator.raw_sha256, self.raw_sha256, "raw hash"),
                (chunk.locator.enhanced_path, self.enhanced_path, "enhanced path"),
                (
                    chunk.locator.enhanced_sha256,
                    self.enhanced_sha256,
                    "enhanced hash",
                ),
                (chunk.locator.docling_path, self.docling_path, "Docling path"),
                (chunk.locator.docling_sha256, self.docling_sha256, "Docling hash"),
                (
                    chunk.locator.fact_document_path,
                    self.fact_document_path,
                    "fact document path",
                ),
            )
            for actual, expected, label in identity_pairs:
                if actual != expected:
                    raise ValueError(
                        f"chunk locator {label} does not match fact document"
                    )
            if chunk.ordinal in ordinals:
                raise ValueError(f"duplicate chunk ordinal: {chunk.ordinal}")
            if chunk.chunk_id in chunk_ids:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            ordinals.add(chunk.ordinal)
            chunk_ids.add(chunk.chunk_id)
        return self


class FactBuildManifestEntry(ContractModel):
    document_id: NonEmpty
    raw_path: NonEmpty
    raw_sha256: Sha256
    enhanced_path: NonEmpty
    enhanced_sha256: Sha256
    fact_document_path: NonEmpty
    fact_document_sha256: Sha256
    docling_path: NonEmpty
    docling_sha256: Sha256
    chunk_count: Annotated[int, Field(ge=0)]


class FactBuildManifest(ContractModel):
    schema_version: Literal["1"] = "1"
    build_id: NonEmpty
    created_at: datetime
    config_version: NonEmpty = "1"
    document_count: Annotated[int, Field(ge=0)]
    chunk_count: Annotated[int, Field(ge=0)]
    documents: tuple[FactBuildManifestEntry, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> FactBuildManifest:
        if self.document_count != len(self.documents):
            raise ValueError("document_count does not match documents")
        if self.chunk_count != sum(item.chunk_count for item in self.documents):
            raise ValueError("chunk_count does not match document entries")
        return self


def canonical_chunk_sha256(
    *,
    raw_text: str,
    contextualized_text: str,
    headings: Sequence[str],
    images: Sequence[ImageReference],
    self_refs: Sequence[str],
) -> str:
    """Hash the exact canonical chunk payload shared by build and read paths."""

    payload = (
        json.dumps(
            {
                "contextualized_text": contextualized_text,
                "headings": list(headings),
                "images": [image.model_dump(mode="json") for image in images],
                "raw_text": raw_text,
                "self_refs": list(self_refs),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
