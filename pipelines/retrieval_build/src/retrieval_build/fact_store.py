from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ai_pm_contracts import (
    DocumentChunkLocator,
    Evidence,
    FactBuildManifest,
    FactBuildManifestEntry,
    FactDocument,
    canonical_chunk_sha256,
)

from .errors import RetrievalError


class FactStoreError(RetrievalError):
    pass


class FactStore:
    """Resolve locators only after revalidating every on-disk fact boundary."""

    def __init__(self, facts_root: str | Path):
        self.root = Path(facts_root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError(f"facts_root is not a directory: {self.root}")
        self.repo_root = self.root.parent.resolve(strict=True)
        self.manifest_path = self.root / "manifests" / "build.json"

    def read_source_window(
        self,
        locator: DocumentChunkLocator,
        window: int = 0,
    ) -> tuple[Evidence, ...]:
        """Return the parent HybridChunker block plus N neighbors per side."""

        if isinstance(window, bool) or not isinstance(window, int) or window < 0:
            raise FactStoreError(
                "INVALID_WINDOW", "window must be a non-negative integer"
            )

        self._validate_locator_paths(locator)
        entry = self._load_manifest_entry(locator)
        document = self._load_document(entry)
        self._validate_identity(document, entry, locator)
        self._verify_disk_artifacts(entry, locator)

        ordered = sorted(document.chunks, key=lambda chunk: chunk.ordinal)
        target_index = next(
            (
                index
                for index, chunk in enumerate(ordered)
                if chunk.ordinal == locator.ordinal
                and chunk.chunk_id == locator.chunk_id
            ),
            None,
        )
        if target_index is None:
            raise FactStoreError(
                "STALE_LOCATOR",
                f"chunk {locator.chunk_id!r} at ordinal {locator.ordinal} no longer exists",
            )

        target = ordered[target_index]
        if target.locator != locator:
            raise FactStoreError(
                "STALE_LOCATOR",
                f"stored locator changed for {locator.chunk_id!r}",
            )
        actual_chunk_sha256 = canonical_chunk_sha256(
            raw_text=target.raw_text,
            contextualized_text=target.contextualized_text,
            headings=target.headings,
            images=target.images,
            self_refs=target.doc_item_refs,
        )
        if actual_chunk_sha256 != locator.chunk_sha256:
            raise FactStoreError(
                "STALE_LOCATOR",
                f"canonical chunk content changed for {locator.chunk_id!r}",
            )

        start = max(0, target_index - window)
        stop = min(len(ordered), target_index + window + 1)
        result: list[Evidence] = []
        for chunk in ordered[start:stop]:
            metadata = dict(document.metadata)
            metadata.update(chunk.metadata)
            result.append(
                Evidence(
                    evidence_id=self._evidence_id(chunk.locator),
                    evidence_kind="source_chunk",
                    locator=chunk.locator,
                    text=chunk.raw_text,
                    title=document.title,
                    headings=chunk.headings,
                    metadata=metadata,
                    doc_item_refs=chunk.doc_item_refs,
                )
            )
        return tuple(result)

    def _load_manifest_entry(
        self, locator: DocumentChunkLocator
    ) -> FactBuildManifestEntry:
        try:
            manifest = FactBuildManifest.model_validate_json(
                self.manifest_path.read_bytes()
            )
        except FileNotFoundError as exc:
            raise FactStoreError(
                "LOCATOR_NOT_FOUND", "facts build manifest is missing"
            ) from exc
        except (OSError, ValueError) as exc:
            raise FactStoreError(
                "INVALID_FACT_ARTIFACT", "facts build manifest is invalid"
            ) from exc
        entry = next(
            (
                item
                for item in manifest.documents
                if item.document_id == locator.document_id
            ),
            None,
        )
        if entry is None:
            raise FactStoreError(
                "STALE_LOCATOR",
                f"document {locator.document_id!r} is absent from the facts manifest",
            )
        return entry

    def _load_document(self, entry: FactBuildManifestEntry) -> FactDocument:
        path = self._resolve_facts_path(entry.fact_document_path)
        payload = self._read_verified(
            path,
            entry.fact_document_sha256,
            label="fact document",
        )
        try:
            return FactDocument.model_validate_json(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise FactStoreError(
                "INVALID_FACT_ARTIFACT",
                f"invalid fact document: {entry.fact_document_path}",
            ) from exc

    def _verify_disk_artifacts(
        self,
        entry: FactBuildManifestEntry,
        locator: DocumentChunkLocator,
    ) -> None:
        # Each potentially large artifact is read and hashed exactly once per
        # source-window call, independent of the requested neighbor count.
        checks = (
            (
                self._resolve_repo_path(locator.raw_path),
                locator.raw_sha256,
                entry.raw_sha256,
                "raw source",
            ),
            (
                self._resolve_repo_path(locator.enhanced_path),
                locator.enhanced_sha256,
                entry.enhanced_sha256,
                "enhanced source",
            ),
            (
                self._resolve_facts_path(locator.docling_path),
                locator.docling_sha256,
                entry.docling_sha256,
                "Docling document",
            ),
        )
        for path, locator_hash, manifest_hash, label in checks:
            if locator_hash != manifest_hash:
                raise FactStoreError(
                    "STALE_LOCATOR", f"{label} hash differs from the facts manifest"
                )
            self._read_verified(path, locator_hash, label=label)

    @staticmethod
    def _validate_identity(
        document: FactDocument,
        entry: FactBuildManifestEntry,
        locator: DocumentChunkLocator,
    ) -> None:
        identity = (
            (
                document.document_id,
                entry.document_id,
                locator.document_id,
                "document_id",
            ),
            (document.raw_path, entry.raw_path, locator.raw_path, "raw path"),
            (document.raw_sha256, entry.raw_sha256, locator.raw_sha256, "raw hash"),
            (
                document.enhanced_path,
                entry.enhanced_path,
                locator.enhanced_path,
                "enhanced path",
            ),
            (
                document.enhanced_sha256,
                entry.enhanced_sha256,
                locator.enhanced_sha256,
                "enhanced hash",
            ),
            (
                document.docling_path,
                entry.docling_path,
                locator.docling_path,
                "Docling path",
            ),
            (
                document.docling_sha256,
                entry.docling_sha256,
                locator.docling_sha256,
                "Docling hash",
            ),
            (
                document.fact_document_path,
                entry.fact_document_path,
                locator.fact_document_path,
                "fact document path",
            ),
        )
        for document_value, manifest_value, locator_value, label in identity:
            if not document_value == manifest_value == locator_value:
                raise FactStoreError(
                    "STALE_LOCATOR", f"{label} no longer matches locator"
                )
        if document.config_version != locator.config_version:
            raise FactStoreError(
                "STALE_LOCATOR", "config version no longer matches locator"
            )

    def _resolve_repo_path(self, relative_path: str) -> Path:
        return self._resolve_under_root(self.repo_root, relative_path)

    def _resolve_facts_path(self, relative_path: str) -> Path:
        return self._resolve_under_root(self.root, relative_path)

    def _validate_locator_paths(self, locator: DocumentChunkLocator) -> None:
        self._resolve_repo_path(locator.raw_path)
        self._resolve_repo_path(locator.enhanced_path)
        self._resolve_facts_path(locator.docling_path)
        self._resolve_facts_path(locator.fact_document_path)

    @staticmethod
    def _resolve_under_root(root: Path, relative_path: str) -> Path:
        candidate_input = Path(relative_path)
        if candidate_input.is_absolute():
            raise FactStoreError("INVALID_LOCATOR", "absolute paths are not allowed")
        candidate = (root / candidate_input).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise FactStoreError(
                "INVALID_LOCATOR", "locator path escapes its configured root"
            ) from exc
        return candidate

    @staticmethod
    def _read_verified(path: Path, expected_sha256: str, *, label: str) -> bytes:
        try:
            payload = path.read_bytes()
        except FileNotFoundError as exc:
            raise FactStoreError(
                "LOCATOR_NOT_FOUND", f"{label} not found: {path}"
            ) from exc
        except OSError as exc:
            raise FactStoreError(
                "INVALID_FACT_ARTIFACT", f"could not read {label}: {path}"
            ) from exc
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise FactStoreError(
                "STALE_LOCATOR", f"{label} hash no longer matches locator"
            )
        return payload

    @staticmethod
    def _evidence_id(locator: DocumentChunkLocator) -> str:
        return f"source:{locator.document_id}:{locator.chunk_id}:{locator.chunk_sha256[:12]}"
