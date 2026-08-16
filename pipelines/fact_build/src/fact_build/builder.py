from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from ai_pm_contracts import (
    DocumentChunkLocator,
    DocumentSegment,
    FactBuildManifest,
    FactBuildManifestEntry,
    FactChunkRecord,
    FactDocument,
    ImageReference,
    canonical_chunk_sha256,
)
from docling.datamodel.backend_options import MarkdownBackendOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import DocumentConverter, MarkdownFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import (
    CodeItem,
    DocItemLabel,
    DoclingDocument,
    ImageRef,
    SectionHeaderItem,
    Size,
    TitleItem,
)
from markdown_it import MarkdownIt
from transformers import AutoTokenizer

from .config import FactBuildConfig, FactBuildConfigError

ProgressCallback = Callable[[str], None]


class FactBuildError(RuntimeError):
    """Raised when a source cannot pass the fact-layer quality gate."""


@dataclass(frozen=True)
class SourceRecord:
    document_id: str
    document_type: str
    metadata: dict[str, Any]
    raw_path: str
    raw_sha256: str
    enhanced_path: str
    enhanced_sha256: str


@dataclass(frozen=True)
class MarkdownImage:
    url: str
    caption: str | None


@dataclass(frozen=True)
class BuiltDocument:
    document: FactDocument
    docling_bytes: bytes
    fact_bytes: bytes
    manifest_entry: FactBuildManifestEntry
    metrics: dict[str, Any]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise FactBuildError(f"frontmatter contains a non-JSON value: {type(value).__name__}")


def _frontmatter(raw_bytes: bytes, source_path: str) -> dict[str, Any]:
    text = raw_bytes.decode("utf-8-sig")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise FactBuildError(f"raw source has no opening frontmatter fence: {source_path}")
    closing = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        raise FactBuildError(f"raw source has no closing frontmatter fence: {source_path}")
    loaded = yaml.safe_load("\n".join(lines[1:closing]))
    if not isinstance(loaded, dict):
        raise FactBuildError(f"raw frontmatter is not a mapping: {source_path}")
    normalized = _json_value(loaded)
    if not isinstance(normalized, dict):
        raise FactBuildError(f"raw frontmatter is not JSON-compatible: {source_path}")
    return normalized


def _markdown_images(text: str) -> tuple[MarkdownImage, ...]:
    tokens = MarkdownIt("commonmark").parse(text)
    images: list[MarkdownImage] = []

    def visit(items: Iterable[Any]) -> None:
        for token in items:
            if token.type == "image":
                url = token.attrGet("src")
                if not url:
                    raise FactBuildError("Markdown image has no source URL")
                caption = token.content.strip() or None
                images.append(MarkdownImage(url=str(url), caption=caption))
            if token.children:
                visit(token.children)

    visit(tokens)
    return tuple(images)


def _document_paths(config: FactBuildConfig, document_id: str) -> tuple[Path, Path, str, str]:
    source_relative = Path(document_id)
    docling_name = source_relative.with_suffix(".docling.json")
    chunks_name = source_relative.with_suffix(".chunks.json")
    docling_absolute = config.output_path("documents_root") / docling_name
    chunks_absolute = config.output_path("chunks_root") / chunks_name
    return (
        docling_absolute,
        chunks_absolute,
        config.facts_relative(docling_absolute),
        config.facts_relative(chunks_absolute),
    )


def _safe_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _assert_chunk_integrity(chunk: FactChunkRecord) -> None:
    canonical_hash = canonical_chunk_sha256(
        raw_text=chunk.raw_text,
        contextualized_text=chunk.contextualized_text,
        headings=chunk.headings,
        images=chunk.images,
        self_refs=chunk.doc_item_refs,
    )
    if chunk.locator.chunk_sha256 != canonical_hash:
        raise FactBuildError(f"canonical chunk hash mismatch: {chunk.chunk_id}")


class FactBuilder:
    """Builds locally reproducible Docling documents and HybridChunker facts."""

    def __init__(
        self,
        config: FactBuildConfig,
        *,
        progress: ProgressCallback | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.config = config
        self.progress = progress or (lambda _message: None)
        self._assert_runtime_versions()

        backend_options = MarkdownBackendOptions(
            enable_remote_fetch=False,
            enable_local_fetch=False,
            fetch_images=False,
        )
        format_option = MarkdownFormatOption(
            pipeline_cls=SimplePipeline,
            backend_options=backend_options,
        )
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.MD],
            format_options={InputFormat.MD: format_option},
        )

        chunking = config.data["chunking"]
        hugging_face_tokenizer = AutoTokenizer.from_pretrained(
            chunking["tokenizer_model"],
            revision=chunking["tokenizer_revision"],
            local_files_only=local_files_only,
        )
        hugging_face_tokenizer.model_max_length = int(chunking["max_tokens"])
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=hugging_face_tokenizer,
            max_tokens=int(chunking["max_tokens"]),
        )
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=bool(chunking["merge_peers"]),
        )
        self.records = self._load_source_records()

    def _assert_runtime_versions(self) -> None:
        required = {
            "docling-slim": self.config.data["docling"]["version"],
            "docling-core": self.config.data["docling"]["core_version"],
        }
        for distribution, expected in required.items():
            actual = version(distribution)
            if actual != expected:
                raise FactBuildError(
                    f"{distribution} runtime is {actual}; required version is {expected}"
                )

    def _load_source_records(self) -> dict[str, SourceRecord]:
        provenance_path = self.config.input_path("enhanced_provenance")
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        documents = payload.get("documents")
        if not isinstance(documents, list):
            raise FactBuildError("enhanced provenance manifest has no documents list")
        if len(documents) != self.config.expected_document_count:
            raise FactBuildError(
                "enhanced provenance document count is "
                f"{len(documents)}, expected {self.config.expected_document_count}"
            )

        records: dict[str, SourceRecord] = {}
        for item in documents:
            document_id = str(item["document_id"])
            if document_id in records:
                raise FactBuildError(f"duplicate provenance document_id: {document_id}")
            record = SourceRecord(
                document_id=document_id,
                document_type=str(item["document_type"]),
                metadata=_json_value(item["metadata"]),
                raw_path=str(item["raw"]["path"]),
                raw_sha256=str(item["raw"]["sha256"]),
                enhanced_path=str(item["enhanced"]["path"]),
                enhanced_sha256=str(item["enhanced"]["sha256"]),
            )
            expected_enhanced = (
                Path(self.config.data["input"]["enhanced_root"]) / document_id
            ).as_posix()
            expected_raw = (Path(self.config.data["input"]["raw_root"]) / document_id).as_posix()
            if record.enhanced_path != expected_enhanced or record.raw_path != expected_raw:
                raise FactBuildError(f"provenance path mismatch for {document_id}")
            records[document_id] = record
        return records

    def _verified_source(self, record: SourceRecord) -> tuple[bytes, bytes, dict[str, Any]]:
        raw_path = self.config.repo_path(record.raw_path)
        enhanced_path = self.config.repo_path(record.enhanced_path)
        raw_bytes = raw_path.read_bytes()
        enhanced_bytes = enhanced_path.read_bytes()
        if _sha256(raw_bytes) != record.raw_sha256:
            raise FactBuildError(f"raw SHA-256 mismatch: {record.document_id}")
        if _sha256(enhanced_bytes) != record.enhanced_sha256:
            raise FactBuildError(f"enhanced SHA-256 mismatch: {record.document_id}")
        metadata = _frontmatter(raw_bytes, record.raw_path)
        if metadata != record.metadata:
            raise FactBuildError(
                f"raw frontmatter does not match provenance metadata: {record.document_id}"
            )
        if metadata.get("type") != record.document_type:
            raise FactBuildError(f"document type mismatch: {record.document_id}")
        return raw_bytes, enhanced_bytes, metadata

    @staticmethod
    def _enrich_images(
        document: DoclingDocument,
        source_images: tuple[MarkdownImage, ...],
        document_id: str,
    ) -> tuple[ImageReference, ...]:
        if len(document.pictures) != len(source_images):
            raise FactBuildError(
                f"image structure mismatch for {document_id}: Markdown has "
                f"{len(source_images)}, Docling has {len(document.pictures)}"
            )

        references: list[ImageReference] = []
        for ordinal, (picture, source) in enumerate(zip(document.pictures, source_images)):
            suffix = Path(urlsplit(source.url).path).suffix
            mimetype = mimetypes.types_map.get(suffix.lower(), "application/octet-stream")
            picture.image = ImageRef(
                mimetype=mimetype,
                dpi=72,
                size=Size(),
                uri=source.url,
            )
            existing_captions = [
                reference.resolve(document).text for reference in picture.captions
            ]
            expected_captions = [source.caption] if source.caption else []
            if existing_captions not in ([], expected_captions):
                raise FactBuildError(
                    f"Docling produced unexpected image captions for {document_id}#image={ordinal}"
                )
            if source.caption and not existing_captions:
                caption = document.add_text(
                    label=DocItemLabel.CAPTION,
                    text=source.caption,
                    parent=picture,
                )
                picture.captions.append(caption.get_ref())
            references.append(
                ImageReference(
                    image_id=f"{document_id}#image={ordinal:06d}",
                    self_ref=str(picture.self_ref),
                    url=source.url,
                    caption=source.caption,
                )
            )
        return tuple(references)

    @staticmethod
    def _associate_images(
        document: DoclingDocument,
        chunk_refs: list[tuple[str, ...]],
        images: tuple[ImageReference, ...],
    ) -> list[tuple[ImageReference, ...]]:
        associations: list[list[ImageReference]] = [[] for _ in chunk_refs]
        if not images:
            return [tuple(items) for items in associations]
        if not chunk_refs:
            raise FactBuildError("document has images but no chunks")

        item_order = {
            str(item.self_ref): index
            for index, (item, _level) in enumerate(
                document.iterate_items(with_groups=False, traverse_pictures=True)
            )
        }
        chunk_positions = [
            tuple(item_order[ref] for ref in refs if ref in item_order) for refs in chunk_refs
        ]
        if any(not positions for positions in chunk_positions):
            raise FactBuildError("chunk contains no resolvable Docling item references")

        for image in images:
            if image.self_ref not in item_order:
                raise FactBuildError(f"image self_ref is unresolved: {image.self_ref}")
            image_position = item_order[image.self_ref]
            following: list[tuple[int, int]] = []
            preceding: list[tuple[int, int]] = []
            for index, positions in enumerate(chunk_positions):
                after = [position - image_position for position in positions if position >= image_position]
                before = [image_position - position for position in positions if position < image_position]
                if after:
                    following.append((min(after), index))
                elif before:
                    preceding.append((min(before), index))
            if following:
                target = min(following)[1]
            elif preceding:
                target = min(preceding)[1]
            else:
                raise FactBuildError(f"cannot associate image with a chunk: {image.image_id}")
            associations[target].append(image)
        return [tuple(items) for items in associations]

    def convert_one(self, document_id: str) -> BuiltDocument:
        started = time.perf_counter()
        try:
            record = self.records[document_id]
        except KeyError as exc:
            raise FactBuildError(f"document is absent from provenance: {document_id}") from exc
        _raw_bytes, enhanced_bytes, metadata = self._verified_source(record)
        enhanced_text = enhanced_bytes.decode("utf-8-sig")
        source_images = _markdown_images(enhanced_text)

        result = self.converter.convert(self.config.repo_path(record.enhanced_path))
        if result.status is not ConversionStatus.SUCCESS:
            raise FactBuildError(f"Docling conversion status is {result.status}: {document_id}")
        docling_document = result.document
        image_references = self._enrich_images(
            docling_document,
            source_images,
            document_id,
        )

        docling_payload = docling_document.model_dump(mode="json", by_alias=True)
        docling_bytes = _json_bytes(docling_payload)
        roundtripped = DoclingDocument.model_validate_json(docling_bytes)
        if roundtripped != docling_document:
            raise FactBuildError(f"Docling JSON roundtrip mismatch: {document_id}")
        docling_sha256 = _sha256(docling_bytes)

        chunks = list(self.chunker.chunk(docling_document))
        if not chunks:
            raise FactBuildError(f"HybridChunker produced no chunks: {document_id}")
        raw_texts = [chunk.text.strip() for chunk in chunks]
        contextualized_texts = [self.chunker.contextualize(chunk).strip() for chunk in chunks]
        chunk_refs = [
            tuple(str(item.self_ref) for item in chunk.meta.doc_items) for chunk in chunks
        ]
        headings = [tuple(str(value) for value in chunk.meta.headings) for chunk in chunks]
        image_associations = self._associate_images(
            docling_document,
            chunk_refs,
            image_references,
        )

        _docling_absolute, _chunks_absolute, docling_path, fact_document_path = (
            _document_paths(self.config, document_id)
        )
        fact_chunks: list[FactChunkRecord] = []
        segment_counts: dict[tuple[str, ...], int] = {}
        for ordinal, (raw_text, contextualized, refs, chunk_headings, chunk_images) in enumerate(
            zip(raw_texts, contextualized_texts, chunk_refs, headings, image_associations)
        ):
            if not raw_text or not contextualized:
                raise FactBuildError(f"empty chunk text at ordinal {ordinal}: {document_id}")
            if not refs:
                raise FactBuildError(f"chunk has no Docling refs at ordinal {ordinal}: {document_id}")
            token_count = self.tokenizer.count_tokens(contextualized)
            if token_count < 1 or token_count > self.config.max_tokens:
                raise FactBuildError(
                    f"chunk token count {token_count} is outside 1..{self.config.max_tokens}: "
                    f"{document_id}#{ordinal}"
                )
            segment_ordinal = segment_counts.get(refs, 0)
            segment_counts[refs] = segment_ordinal + 1
            chunk_id = f"{document_id}#chunk={ordinal:06d}"
            chunk_sha256 = canonical_chunk_sha256(
                raw_text=raw_text,
                contextualized_text=contextualized,
                headings=chunk_headings,
                images=chunk_images,
                self_refs=refs,
            )
            locator = DocumentChunkLocator(
                config_version=self.config.config_version,
                document_id=document_id,
                chunk_id=chunk_id,
                ordinal=ordinal,
                segment_ordinal=segment_ordinal,
                fact_document_path=fact_document_path,
                raw_path=record.raw_path,
                raw_sha256=record.raw_sha256,
                enhanced_path=record.enhanced_path,
                enhanced_sha256=record.enhanced_sha256,
                docling_path=docling_path,
                docling_sha256=docling_sha256,
                chunk_sha256=chunk_sha256,
                segment=DocumentSegment(
                    first_self_ref=refs[0],
                    last_self_ref=refs[-1],
                    item_count=len(refs),
                ),
                self_refs=refs,
                headings=chunk_headings,
                images=chunk_images,
            )
            fact_chunks.append(
                FactChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    ordinal=ordinal,
                    raw_text=raw_text,
                    contextualized_text=contextualized,
                    token_count=token_count,
                    headings=chunk_headings,
                    metadata={},
                    doc_item_refs=refs,
                    images=chunk_images,
                    locator=locator,
                )
            )

        fact_document = FactDocument(
            config_version=self.config.config_version,
            document_id=document_id,
            raw_path=record.raw_path,
            raw_sha256=record.raw_sha256,
            enhanced_path=record.enhanced_path,
            enhanced_sha256=record.enhanced_sha256,
            docling_path=docling_path,
            docling_sha256=docling_sha256,
            fact_document_path=fact_document_path,
            title=str(metadata.get("title")) if metadata.get("title") is not None else None,
            document_type=record.document_type,
            metadata=metadata,
            chunks=tuple(fact_chunks),
        )
        fact_bytes = _json_bytes(fact_document.model_dump(mode="json"))
        if FactDocument.model_validate_json(fact_bytes) != fact_document:
            raise FactBuildError(f"FactDocument JSON roundtrip mismatch: {document_id}")

        metrics = self._validate_document(
            fact_document,
            docling_document,
            source_images,
            enhanced_text,
        )
        metrics["duration_seconds"] = round(time.perf_counter() - started, 3)
        entry = FactBuildManifestEntry(
            document_id=document_id,
            raw_path=record.raw_path,
            raw_sha256=record.raw_sha256,
            enhanced_path=record.enhanced_path,
            enhanced_sha256=record.enhanced_sha256,
            fact_document_path=fact_document_path,
            fact_document_sha256=_sha256(fact_bytes),
            docling_path=docling_path,
            docling_sha256=docling_sha256,
            chunk_count=len(fact_chunks),
        )
        return BuiltDocument(
            document=fact_document,
            docling_bytes=docling_bytes,
            fact_bytes=fact_bytes,
            manifest_entry=entry,
            metrics=metrics,
        )

    def _validate_document(
        self,
        fact_document: FactDocument,
        docling_document: DoclingDocument,
        source_images: tuple[MarkdownImage, ...],
        enhanced_text: str,
    ) -> dict[str, Any]:
        document_id = fact_document.document_id
        all_items = tuple(
            item
            for item, _level in docling_document.iterate_items(
                with_groups=False,
                traverse_pictures=True,
            )
        )
        self_refs = {str(item.self_ref) for item in all_items}
        title_count = sum(isinstance(item, TitleItem) for item in all_items)
        section_headers = [item for item in all_items if isinstance(item, SectionHeaderItem)]
        code_count = sum(isinstance(item, CodeItem) for item in all_items)
        if title_count < 1:
            raise FactBuildError(f"document has no Docling title: {document_id}")

        expected_ordinals = list(range(len(fact_document.chunks)))
        if [chunk.ordinal for chunk in fact_document.chunks] != expected_ordinals:
            raise FactBuildError(f"chunk ordinals are not contiguous: {document_id}")
        for chunk in fact_document.chunks:
            if any(ref not in self_refs for ref in chunk.doc_item_refs):
                raise FactBuildError(f"chunk contains unresolved Docling refs: {chunk.chunk_id}")
            if chunk.token_count != self.tokenizer.count_tokens(chunk.contextualized_text):
                raise FactBuildError(f"stored token count mismatch: {chunk.chunk_id}")
            if chunk.token_count > self.config.max_tokens:
                raise FactBuildError(f"chunk exceeds configured token limit: {chunk.chunk_id}")
            _assert_chunk_integrity(chunk)

        pictures = tuple(docling_document.pictures)
        if len(pictures) != len(source_images):
            raise FactBuildError(f"image count changed during validation: {document_id}")
        fact_images = [image for chunk in fact_document.chunks for image in chunk.images]
        if len(fact_images) != len(source_images):
            raise FactBuildError(f"not every image is attached to one chunk: {document_id}")
        if len({image.image_id for image in fact_images}) != len(fact_images):
            raise FactBuildError(f"an image is attached to multiple chunks: {document_id}")
        for picture, source, fact_image in zip(pictures, source_images, fact_images):
            if picture.image is None or str(picture.image.uri) != source.url:
                raise FactBuildError(f"Docling image URL was not preserved: {document_id}")
            captions = [reference.resolve(docling_document).text for reference in picture.captions]
            expected_captions = [source.caption] if source.caption else []
            if captions != expected_captions:
                raise FactBuildError(f"Docling image caption was not preserved: {document_id}")
            if fact_image.url != source.url or fact_image.caption != source.caption:
                raise FactBuildError(f"fact image metadata mismatch: {document_id}")

        checks: dict[str, bool] = {
            "contract_roundtrip": True,
            "docling_roundtrip": True,
            "image_urls_and_captions": True,
            "ordinals_contiguous": True,
            "refs_resolve": True,
            "tokens_within_limit": True,
        }
        if document_id.endswith("get-perplexity-pro-free-for-1-year.md"):
            checks["short_without_section_heading"] = len(section_headers) == 0
        elif document_id.endswith("how-to-kickstart-and-scale-a-consumer-business.md"):
            checks["large_complex_structure"] = (
                len(fact_document.chunks) >= 10
                and len(section_headers) >= 10
                and len(source_images) >= 10
            )
        elif document_id.endswith("how-to-measure-cohort-retention.md"):
            checks["code_items_preserved"] = code_count >= 1
        elif document_id.endswith("how-to-run-seo-experiments.md"):
            checks["native_html_preserved"] = (
                "<title>Figma" in enhanced_text
                and any("Figma - iOS & iPadOS 14 UI Kit" in getattr(item, "text", "") for item in all_items)
            )
        elif document_id.endswith("03-podcasts/eric-ries.md"):
            levels = {header.level for header in section_headers}
            checks["podcast_heading_hierarchy"] = 1 in levels and 2 in levels

        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            raise FactBuildError(
                f"quality checks failed for {document_id}: {', '.join(failed_checks)}"
            )
        return {
            "checks": checks,
            "chunk_count": len(fact_document.chunks),
            "code_item_count": code_count,
            "image_count": len(source_images),
            "max_chunk_tokens": max(chunk.token_count for chunk in fact_document.chunks),
            "section_header_count": len(section_headers),
            "section_header_levels": sorted({header.level for header in section_headers}),
            "title_count": title_count,
        }

    def _write_quarantine(self, document_id: str, exc: Exception, stage: str) -> None:
        safe_name = document_id.replace("/", "__").replace("\\", "__")
        path = self.config.output_path("quarantine_root") / f"{safe_name}.json"
        payload = {
            "config_version": self.config.config_version,
            "document_id": document_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "stage": stage,
        }
        _safe_write(path, _json_bytes(payload))

    def clear_quarantine(self) -> None:
        root = self.config.output_path("quarantine_root")
        root.mkdir(parents=True, exist_ok=True)
        for path in root.glob("*.json"):
            if path.is_file():
                path.unlink()

    def run_pilot(self) -> dict[str, Any]:
        self.clear_quarantine()
        entries: list[dict[str, Any]] = []
        passed = True
        for index, document_id in enumerate(self.config.pilot_documents, start=1):
            self.progress(f"pilot {index}/5: {document_id}")
            try:
                built = self.convert_one(document_id)
                entries.append(
                    {
                        "document_id": document_id,
                        "status": "passed",
                        **built.metrics,
                    }
                )
            # The pilot is a batch isolation boundary: quarantine every document-local
            # conversion failure and continue reporting the remaining representatives.
            except Exception as exc:  # noqa: BLE001
                passed = False
                self._write_quarantine(document_id, exc, "pilot")
                entries.append(
                    {
                        "document_id": document_id,
                        "status": "failed",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )

        report = {
            "schema_version": "1",
            "config_version": self.config.config_version,
            "config_sha256": self.config.sha256,
            "created_at": datetime.now(UTC).isoformat(),
            "document_count": len(entries),
            "passed": passed,
            "passed_count": sum(entry["status"] == "passed" for entry in entries),
            "failed_count": sum(entry["status"] == "failed" for entry in entries),
            "documents": entries,
        }
        _safe_write(self.config.output_path("pilot_report"), _json_bytes(report))
        if not passed and self.config.data["pilot"].get("require_all_valid", True):
            raise FactBuildError("pilot gate failed; full conversion was not started")
        return report

    def run_full_build(self, *, require_pilot: bool = True) -> FactBuildManifest:
        if require_pilot:
            self.run_pilot()
        self.clear_quarantine()

        facts_root = (self.config.repo_root / "facts").resolve(strict=True)
        staging = facts_root / f".fact-build-staging-{uuid.uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)
        (staging / self.config.facts_relative(self.config.output_path("assets_root"))).mkdir(
            parents=True,
            exist_ok=False,
        )
        entries: list[FactBuildManifestEntry] = []
        total_chunks = 0
        documents = sorted(self.records)
        try:
            for index, document_id in enumerate(documents, start=1):
                if index == 1 or index % 25 == 0 or index == len(documents):
                    self.progress(f"build {index}/{len(documents)}: {document_id}")
                try:
                    built = self.convert_one(document_id)
                except Exception as exc:
                    self._write_quarantine(document_id, exc, "full")
                    raise
                docling_relative = built.document.docling_path
                fact_relative = built.document.chunks[0].locator.fact_document_path
                _safe_write(staging / docling_relative, built.docling_bytes)
                _safe_write(staging / fact_relative, built.fact_bytes)
                entries.append(built.manifest_entry)
                total_chunks += len(built.document.chunks)

            build_material = _json_bytes(
                {
                    "config_sha256": self.config.sha256,
                    "documents": [
                        {
                            "document_id": entry.document_id,
                            "raw_sha256": entry.raw_sha256,
                            "enhanced_sha256": entry.enhanced_sha256,
                            "docling_sha256": entry.docling_sha256,
                            "fact_document_sha256": entry.fact_document_sha256,
                        }
                        for entry in entries
                    ],
                }
            )
            manifest = FactBuildManifest(
                build_id=f"{self.config.config_version}-{_sha256(build_material)[:20]}",
                created_at=datetime.now(UTC),
                config_version=self.config.config_version,
                document_count=len(entries),
                chunk_count=total_chunks,
                documents=tuple(entries),
            )
            if manifest.document_count != self.config.expected_document_count:
                raise FactBuildError(
                    f"full build has {manifest.document_count} documents, expected "
                    f"{self.config.expected_document_count}"
                )

            for output_key in ("documents_root", "chunks_root", "assets_root"):
                target = self.config.output_path(output_key)
                staged = staging / self.config.facts_relative(target)
                if target.exists():
                    shutil.rmtree(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
            manifest_bytes = _json_bytes(manifest.model_dump(mode="json"))
            if FactBuildManifest.model_validate_json(manifest_bytes) != manifest:
                raise FactBuildError("FactBuildManifest JSON roundtrip mismatch")
            _safe_write(self.config.output_path("build_manifest"), manifest_bytes)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def validate_published(self) -> dict[str, Any]:
        manifest_path = self.config.output_path("build_manifest")
        manifest_bytes = manifest_path.read_bytes()
        manifest = FactBuildManifest.model_validate_json(manifest_bytes)
        if manifest.document_count != self.config.expected_document_count:
            raise FactBuildError("published manifest document count mismatch")
        if manifest.config_version != self.config.config_version:
            raise FactBuildError("published manifest config version mismatch")
        manifest_ids = [entry.document_id for entry in manifest.documents]
        if manifest_ids != sorted(self.records):
            raise FactBuildError("published manifest does not exactly match provenance document IDs")
        fact_paths = [entry.fact_document_path for entry in manifest.documents]
        docling_paths = [entry.docling_path for entry in manifest.documents]
        if len(set(fact_paths)) != len(fact_paths) or len(set(docling_paths)) != len(docling_paths):
            raise FactBuildError("published manifest contains duplicate artifact paths")

        chunk_count = 0
        image_count = 0
        max_chunk_tokens = 0
        for index, entry in enumerate(manifest.documents, start=1):
            if index == 1 or index % 50 == 0 or index == len(manifest.documents):
                self.progress(f"validate {index}/{len(manifest.documents)}: {entry.document_id}")
            record = self.records[entry.document_id]
            expected_docling_absolute, expected_fact_absolute, expected_docling, expected_fact = (
                _document_paths(self.config, entry.document_id)
            )
            expected_identity = (
                (entry.raw_path, record.raw_path, "raw path"),
                (entry.raw_sha256, record.raw_sha256, "raw hash"),
                (entry.enhanced_path, record.enhanced_path, "enhanced path"),
                (entry.enhanced_sha256, record.enhanced_sha256, "enhanced hash"),
                (entry.docling_path, expected_docling, "Docling path"),
                (entry.fact_document_path, expected_fact, "fact document path"),
            )
            for actual, expected, label in expected_identity:
                if actual != expected:
                    raise FactBuildError(
                        f"published manifest {label} mismatch: {entry.document_id}"
                    )

            raw_bytes = self.config.repo_path(entry.raw_path).read_bytes()
            enhanced_bytes = self.config.repo_path(entry.enhanced_path).read_bytes()
            if _sha256(raw_bytes) != entry.raw_sha256:
                raise FactBuildError(f"published raw source hash mismatch: {entry.document_id}")
            if _sha256(enhanced_bytes) != entry.enhanced_sha256:
                raise FactBuildError(f"published enhanced source hash mismatch: {entry.document_id}")

            fact_path = self._resolve_fact_artifact(entry.fact_document_path)
            if fact_path != expected_fact_absolute.resolve(strict=True):
                raise FactBuildError(f"published fact file path mismatch: {entry.document_id}")
            fact_bytes = fact_path.read_bytes()
            if _sha256(fact_bytes) != entry.fact_document_sha256:
                raise FactBuildError(f"published fact document hash mismatch: {entry.document_id}")
            document = FactDocument.model_validate_json(fact_bytes)
            if _json_bytes(document.model_dump(mode="json")) != fact_bytes:
                raise FactBuildError(f"published fact JSON is not canonical: {entry.document_id}")
            if document.document_id != entry.document_id:
                raise FactBuildError(f"manifest document_id mismatch: {entry.document_id}")
            document_identity = (
                (document.raw_path, entry.raw_path, "raw path"),
                (document.raw_sha256, entry.raw_sha256, "raw hash"),
                (document.enhanced_path, entry.enhanced_path, "enhanced path"),
                (document.enhanced_sha256, entry.enhanced_sha256, "enhanced hash"),
                (document.docling_path, entry.docling_path, "Docling path"),
                (document.docling_sha256, entry.docling_sha256, "Docling hash"),
                (document.fact_document_path, entry.fact_document_path, "fact document path"),
            )
            for actual, expected, label in document_identity:
                if actual != expected:
                    raise FactBuildError(
                        f"published FactDocument {label} mismatch: {entry.document_id}"
                    )
            if len(document.chunks) != entry.chunk_count:
                raise FactBuildError(f"manifest chunk count mismatch: {entry.document_id}")
            metadata = _frontmatter(raw_bytes, entry.raw_path)
            if metadata != document.metadata or metadata != record.metadata:
                raise FactBuildError(f"published raw metadata mismatch: {entry.document_id}")
            if document.title != metadata.get("title") or document.document_type != metadata.get("type"):
                raise FactBuildError(f"published title/type mismatch: {entry.document_id}")

            docling_path = self._resolve_fact_artifact(document.docling_path)
            if docling_path != expected_docling_absolute.resolve(strict=True):
                raise FactBuildError(f"published Docling file path mismatch: {entry.document_id}")
            docling_bytes = docling_path.read_bytes()
            if _sha256(docling_bytes) != document.docling_sha256 or _sha256(
                docling_bytes
            ) != entry.docling_sha256:
                raise FactBuildError(f"published Docling hash mismatch: {entry.document_id}")
            docling_document = DoclingDocument.model_validate_json(docling_bytes)
            if _json_bytes(docling_document.model_dump(mode="json", by_alias=True)) != docling_bytes:
                raise FactBuildError(f"published Docling JSON is not canonical: {entry.document_id}")
            known_refs = {
                str(item.self_ref)
                for item, _level in docling_document.iterate_items(
                    with_groups=False,
                    traverse_pictures=True,
                )
            }
            source_images = _markdown_images(enhanced_bytes.decode("utf-8-sig"))
            if len(source_images) != len(docling_document.pictures):
                raise FactBuildError(f"published Docling image count mismatch: {entry.document_id}")
            picture_images: dict[str, tuple[str, str | None]] = {}
            for picture, source_image in zip(docling_document.pictures, source_images):
                captions = [reference.resolve(docling_document).text for reference in picture.captions]
                expected_captions = [source_image.caption] if source_image.caption else []
                if (
                    picture.image is None
                    or str(picture.image.uri) != source_image.url
                    or captions != expected_captions
                ):
                    raise FactBuildError(
                        f"published image URL/caption mismatch: {entry.document_id}"
                    )
                picture_images[str(picture.self_ref)] = (source_image.url, source_image.caption)

            segment_counts: dict[tuple[str, ...], int] = {}
            seen_images: set[str] = set()
            for expected_ordinal, chunk in enumerate(document.chunks):
                if chunk.ordinal != expected_ordinal:
                    raise FactBuildError(f"published ordinal mismatch: {chunk.chunk_id}")
                expected_chunk_id = f"{entry.document_id}#chunk={expected_ordinal:06d}"
                if chunk.chunk_id != expected_chunk_id:
                    raise FactBuildError(f"published stable chunk ID mismatch: {chunk.chunk_id}")
                if chunk.locator.fact_document_path != entry.fact_document_path:
                    raise FactBuildError(f"published fact path mismatch: {chunk.chunk_id}")
                if chunk.locator.docling_path != entry.docling_path:
                    raise FactBuildError(f"published Docling path mismatch: {chunk.chunk_id}")
                locator_identity = (
                    (chunk.locator.raw_path, entry.raw_path),
                    (chunk.locator.raw_sha256, entry.raw_sha256),
                    (chunk.locator.enhanced_path, entry.enhanced_path),
                    (chunk.locator.enhanced_sha256, entry.enhanced_sha256),
                    (chunk.locator.docling_sha256, entry.docling_sha256),
                )
                if any(actual != expected for actual, expected in locator_identity):
                    raise FactBuildError(f"published locator identity mismatch: {chunk.chunk_id}")
                if chunk.headings != chunk.locator.headings or chunk.images != chunk.locator.images:
                    raise FactBuildError(f"published locator content mismatch: {chunk.chunk_id}")
                if any(ref not in known_refs for ref in chunk.doc_item_refs):
                    raise FactBuildError(f"published ref is unresolved: {chunk.chunk_id}")
                expected_segment_ordinal = segment_counts.get(chunk.doc_item_refs, 0)
                if chunk.locator.segment_ordinal != expected_segment_ordinal:
                    raise FactBuildError(f"published segment ordinal mismatch: {chunk.chunk_id}")
                segment_counts[chunk.doc_item_refs] = expected_segment_ordinal + 1
                actual_tokens = self.tokenizer.count_tokens(chunk.contextualized_text)
                if actual_tokens != chunk.token_count or actual_tokens > self.config.max_tokens:
                    raise FactBuildError(f"published token count mismatch: {chunk.chunk_id}")
                _assert_chunk_integrity(chunk)
                for image in chunk.images:
                    if image.image_id in seen_images or image.self_ref not in picture_images:
                        raise FactBuildError(f"published fact image is duplicated/unresolved: {chunk.chunk_id}")
                    if (image.url, image.caption) != picture_images[image.self_ref]:
                        raise FactBuildError(f"published fact image mismatch: {chunk.chunk_id}")
                    seen_images.add(image.image_id)
                max_chunk_tokens = max(max_chunk_tokens, actual_tokens)
                image_count += len(chunk.images)
            if len(seen_images) != len(source_images):
                raise FactBuildError(f"published image attachment count mismatch: {entry.document_id}")
            chunk_count += len(document.chunks)

        if chunk_count != manifest.chunk_count:
            raise FactBuildError("published total chunk count mismatch")
        quarantine_count = len(list(self.config.output_path("quarantine_root").glob("*.json")))
        if quarantine_count:
            raise FactBuildError(f"published build has {quarantine_count} quarantined documents")
        return {
            "build_id": manifest.build_id,
            "document_count": manifest.document_count,
            "chunk_count": chunk_count,
            "image_count": image_count,
            "max_chunk_tokens": max_chunk_tokens,
            "quarantine_count": quarantine_count,
        }

    def _resolve_fact_artifact(self, relative_path: str) -> Path:
        path_input = Path(relative_path)
        if path_input.is_absolute():
            raise FactBuildError(f"absolute fact artifact path is forbidden: {relative_path}")
        facts_root = (self.config.repo_root / "facts").resolve(strict=True)
        candidate = (facts_root / path_input).resolve(strict=True)
        try:
            candidate.relative_to(facts_root)
        except ValueError as exc:
            raise FactBuildError(f"fact artifact path escapes facts root: {relative_path}") from exc
        return candidate


def load_default_config(path: str | Path | None = None) -> FactBuildConfig:
    if path is not None:
        return FactBuildConfig.load(path)
    candidate = Path(__file__).resolve().parents[4] / "config" / "facts" / "build.json"
    if not candidate.exists():
        raise FactBuildConfigError("cannot locate config/facts/build.json; pass --config")
    return FactBuildConfig.load(candidate)
