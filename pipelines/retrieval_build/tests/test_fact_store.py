from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ai_pm_contracts import FactBuildManifest
from conftest import sha
from retrieval_build.fact_store import FactStore, FactStoreError


def test_window_zero_returns_exactly_the_target(
    fact_artifacts: tuple[Path, object],
) -> None:
    root, document = fact_artifacts
    evidence = FactStore(root).read_source_window(document.chunks[1].locator, window=0)
    assert [item.text for item in evidence] == ["one target block"]


def test_window_is_symmetric_clipped_and_never_crosses_document(
    fact_artifacts: tuple[Path, object],
) -> None:
    root, document = fact_artifacts
    evidence = FactStore(root).read_source_window(document.chunks[1].locator, window=1)
    assert [item.text for item in evidence] == [
        "zero block",
        "one target block",
        "two block",
    ]

    first = FactStore(root).read_source_window(document.chunks[0].locator, window=99)
    assert [item.locator.document_id for item in first] == ["alpha"] * 4
    assert [item.text for item in first] == [
        "zero block",
        "one target block",
        "two block",
        "three block",
    ]


def test_hash_mismatch_is_a_stale_locator(fact_artifacts: tuple[Path, object]) -> None:
    root, document = fact_artifacts
    stale = document.chunks[1].locator.model_copy(
        update={"enhanced_sha256": sha("changed")}
    )
    with pytest.raises(FactStoreError) as caught:
        FactStore(root).read_source_window(stale, window=0)
    assert caught.value.code == "STALE_LOCATOR"


def test_chunk_hash_mismatch_is_a_stale_locator(
    fact_artifacts: tuple[Path, object],
) -> None:
    root, document = fact_artifacts
    stale = document.chunks[1].locator.model_copy(
        update={"chunk_sha256": sha("changed")}
    )
    with pytest.raises(FactStoreError) as caught:
        FactStore(root).read_source_window(stale, window=0)
    assert caught.value.code == "STALE_LOCATOR"


def test_any_locator_hash_mismatch_is_stale(
    fact_artifacts: tuple[Path, object],
) -> None:
    root, document = fact_artifacts
    stale = document.chunks[1].locator.model_copy(update={"raw_sha256": sha("old raw")})
    with pytest.raises(FactStoreError) as caught:
        FactStore(root).read_source_window(stale, window=0)
    assert caught.value.code == "STALE_LOCATOR"


def test_fact_path_cannot_escape_root(fact_artifacts: tuple[Path, object]) -> None:
    root, document = fact_artifacts
    malicious = document.chunks[0].locator.model_copy(
        update={"fact_document_path": "../secret.json"}
    )
    with pytest.raises(FactStoreError) as caught:
        FactStore(root).read_source_window(malicious, window=0)
    assert caught.value.code == "INVALID_LOCATOR"


def test_manifest_path_is_restricted_to_active_or_generation_manifests(
    fact_artifacts: tuple[Path, object],
) -> None:
    root, _document = fact_artifacts
    with pytest.raises(ValueError, match="manifest_path"):
        FactStore(root, manifest_path=root.parent / "outside.json")
    with pytest.raises(ValueError, match="manifest_path"):
        FactStore(root, manifest_path=root / "chunks" / "v1" / "alpha.json")


@pytest.mark.parametrize("artifact", ["raw", "enhanced", "docling"])
def test_disk_artifact_tampering_is_stale(
    fact_artifacts: tuple[Path, object], artifact: str
) -> None:
    root, document = fact_artifacts
    locator = document.chunks[1].locator
    if artifact == "raw":
        path = root.parent / locator.raw_path
    elif artifact == "enhanced":
        path = root.parent / locator.enhanced_path
    else:
        path = root / locator.docling_path
    path.write_bytes(b"tampered")
    with pytest.raises(FactStoreError) as caught:
        FactStore(root).read_source_window(locator, window=0)
    assert caught.value.code == "STALE_LOCATOR"


def test_target_chunk_canonical_hash_is_recomputed(
    fact_artifacts: tuple[Path, object],
) -> None:
    root, document = fact_artifacts
    locator = document.chunks[1].locator
    fact_path = root / locator.fact_document_path
    payload = document.model_dump(mode="json")
    payload["chunks"][1]["raw_text"] = "tampered target text"
    fact_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    fact_path.write_bytes(fact_bytes)

    manifest_path = root / "manifests" / "build.json"
    manifest = FactBuildManifest.model_validate_json(manifest_path.read_bytes())
    entries = list(manifest.documents)
    entries[0] = entries[0].model_copy(
        update={"fact_document_sha256": hashlib.sha256(fact_bytes).hexdigest()}
    )
    manifest_path.write_text(
        manifest.model_copy(update={"documents": tuple(entries)}).model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    with pytest.raises(FactStoreError) as caught:
        FactStore(root).read_source_window(locator, window=0)
    assert caught.value.code == "STALE_LOCATOR"
