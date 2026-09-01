from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_pm_contracts import DocumentChunkLocator, Evidence, FactBuildManifest
from retrieval_build import (
    FactStore,
    ValidatedRetrievalBundle,
    load_and_resolve_bundle_manifest,
)


class RetrievalSnapshotChanged(RuntimeError):
    code = "RETRIEVAL_SNAPSHOT_CHANGED"
    retryable = True


@dataclass(frozen=True)
class _ArtifactStamp:
    path: Path
    size: int
    modified_ns: int

    @classmethod
    def capture(cls, path: Path) -> _ArtifactStamp:
        stat = path.stat()
        return cls(path=path, size=stat.st_size, modified_ns=stat.st_mtime_ns)

    def assert_unchanged(self) -> None:
        try:
            stat = self.path.stat()
        except OSError as exc:
            raise RetrievalSnapshotChanged(
                f"validated retrieval artifact disappeared: {self.path}"
            ) from exc
        if stat.st_size != self.size or stat.st_mtime_ns != self.modified_ns:
            raise RetrievalSnapshotChanged(
                f"validated retrieval artifact changed on disk: {self.path}"
            )


@dataclass(frozen=True)
class RetrievalSnapshot:
    """One startup-validated, immutable facts/index generation."""

    bundle: ValidatedRetrievalBundle
    facts_root: Path
    facts_manifest: FactBuildManifest
    _stamps: tuple[_ArtifactStamp, ...]

    @property
    def build_id(self) -> str:
        return self.bundle.manifest.build_id

    @property
    def lexical_path(self) -> Path:
        return self.bundle.lexical_database_path

    @property
    def semantic_path(self) -> Path | None:
        return self.bundle.semantic_database_path

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        *,
        facts_root: str | Path,
    ) -> RetrievalSnapshot:
        bundle = load_and_resolve_bundle_manifest(manifest_path)
        resolved_facts_root = Path(facts_root).resolve(strict=True)
        resolved_fact_manifest = bundle.facts_manifest_path.resolve(strict=True)
        try:
            resolved_fact_manifest.relative_to(resolved_facts_root)
        except ValueError as exc:
            raise ValueError(
                "retrieval bundle facts manifest escapes the configured facts root"
            ) from exc
        facts_manifest = FactBuildManifest.model_validate_json(
            resolved_fact_manifest.read_bytes()
        )
        protected = [
            resolved_fact_manifest,
            bundle.lexical_database_path,
        ]
        if bundle.semantic_database_path is not None:
            protected.append(bundle.semantic_database_path)
        return cls(
            bundle=bundle,
            facts_root=resolved_facts_root,
            facts_manifest=facts_manifest,
            _stamps=tuple(_ArtifactStamp.capture(path) for path in protected),
        )

    def assert_unchanged(self) -> None:
        for stamp in self._stamps:
            stamp.assert_unchanged()


class SnapshotFactStore:
    """Fact reader that fails closed if its validated generation is modified."""

    def __init__(self, snapshot: RetrievalSnapshot):
        self.snapshot = snapshot
        try:
            self.store = FactStore(
                snapshot.facts_root,
                manifest_path=snapshot.bundle.facts_manifest_path,
            )
        except TypeError:
            # Compatibility for old retrieval_build packages during rolling
            # workspace upgrades. The default manifest is safe only when it is
            # exactly the validated one.
            expected = snapshot.facts_root / "manifests" / "build.json"
            if snapshot.bundle.facts_manifest_path != expected:
                raise
            self.store = FactStore(snapshot.facts_root)

    def read_source_window(
        self, locator: DocumentChunkLocator, window: int = 0
    ) -> tuple[Evidence, ...]:
        self.snapshot.assert_unchanged()
        return self.store.read_source_window(locator, window)
