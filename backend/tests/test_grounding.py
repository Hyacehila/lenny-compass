from __future__ import annotations

import pytest
from ai_pm_contracts import DocumentChunkLocator, DocumentSegment, Evidence
from lenny_backend.grounding import bind_cited_evidence, insufficient_evidence_answer
from lenny_backend.providers import ProviderError


def _evidence(document_id: str, chunk_id: str) -> Evidence:
    locator = DocumentChunkLocator(
        document_id=document_id,
        chunk_id=chunk_id,
        ordinal=0,
        segment_ordinal=0,
        fact_document_path=f"documents/v1/{document_id}.json",
        raw_path=f"raw/{document_id}.md",
        raw_sha256="a" * 64,
        enhanced_path=f"enhanced/{document_id}.md",
        enhanced_sha256="b" * 64,
        docling_path=f"documents/v1/{document_id}.docling.json",
        docling_sha256="c" * 64,
        chunk_sha256="d" * 64,
        segment=DocumentSegment(
            first_self_ref="#/texts/0",
            last_self_ref="#/texts/0",
            item_count=1,
        ),
        self_refs=("#/texts/0",),
    )
    return Evidence(
        evidence_id=f"source:{document_id}:{chunk_id}",
        evidence_kind="source_chunk",
        locator=locator,
        text=f"Evidence for {document_id}",
    )


def test_bind_citations_persists_only_evidence_actually_referenced() -> None:
    first = _evidence("one", "chunk-one")
    second = _evidence("two", "chunk-two")

    evidence, claims = bind_cited_evidence(
        f"[evidence:{second.evidence_id}] Supported statement",
        (first, second),
        error_prefix="TEST",
    )

    assert evidence == (second,)
    assert claims[0].text == "Supported statement"
    assert claims[0].evidence_ids == (second.evidence_id,)


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        ("Unsupported statement", "TEST_MISSING_CITATIONS"),
        (
            "Unsupported statement\n[evidence:source:one:chunk-one] Supported",
            "TEST_MISSING_CITATIONS",
        ),
        (
            "# Revenue grew 300%\n[evidence:source:one:chunk-one] Supported",
            "TEST_MISSING_CITATIONS",
        ),
        ("[evidence:unknown] Invented source", "TEST_UNREAD_CITATION"),
    ],
)
def test_bind_citations_rejects_missing_or_unknown_evidence(
    answer: str, code: str
) -> None:
    with pytest.raises(ProviderError) as caught:
        bind_cited_evidence(
            answer, (_evidence("one", "chunk-one"),), error_prefix="TEST"
        )
    assert caught.value.code == code


def test_insufficient_answer_is_deterministic_and_locale_aware() -> None:
    assert (
        insufficient_evidence_answer("这个问题？")
        == "现有证据不足，无法可靠回答这个问题。"
    )
    assert insufficient_evidence_answer("What happened?").startswith(
        "The available evidence is insufficient"
    )
