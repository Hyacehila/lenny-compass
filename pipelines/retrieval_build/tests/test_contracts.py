from __future__ import annotations

import pytest
from ai_pm_contracts import (
    AnswerRequest,
    Evidence,
    ImageObservation,
    ImageReference,
    LlmWikiPageLocator,
    RetrievalHit,
    SseAnswerDeltaEvent,
)
from conftest import make_document
from pydantic import ValidationError


def test_answer_request_defaults_to_direct_and_sse_uses_type_discriminator() -> None:
    assert AnswerRequest(question="What did Lenny say?").mode == "direct"
    assert SseAnswerDeltaEvent(delta="hello").model_dump() == {
        "type": "answer_delta",
        "delta": "hello",
    }


def test_retrieval_hit_forbids_excerpts() -> None:
    locator = make_document("contract", ["source fact"]).chunks[0].locator
    with pytest.raises(ValidationError):
        RetrievalHit(
            provider="keyword",
            rank=1,
            native_score=-1.0,
            locator=locator,
            text="must not cross the retrieval/fact boundary",
        )


def test_reserved_wiki_and_image_contracts_have_stable_explicit_keys() -> None:
    wiki = LlmWikiPageLocator(project_id="ai-pm", path="products/activation")
    assert wiki.model_dump()["project_id"] == "ai-pm"
    assert wiki.model_dump()["path"] == "products/activation"
    image = ImageReference(image_id="chart-1", url="https://example.test/chart.png")
    assert image.url == "https://example.test/chart.png"

    with pytest.raises(ValidationError):
        LlmWikiPageLocator(page_id="unstable", revision="1")


def test_evidence_has_typed_strict_image_observations() -> None:
    image = ImageReference(image_id="chart-1", url="https://example.test/chart.png")
    locator = (
        make_document("vision", ["A chart supports the claim."])
        .chunks[0]
        .locator.model_copy(update={"images": (image,)})
    )
    observation = ImageObservation(
        image_id="chart-1",
        url="https://example.test/chart.png",
        observation="The activation curve rises after onboarding.",
        inspected=True,
    )
    evidence = Evidence(
        evidence_id="source:vision:0",
        evidence_kind="source_chunk",
        locator=locator,
        text="A chart supports the claim.",
        image_observations=(observation,),
    )
    assert evidence.image_observations == (observation,)
    with pytest.raises(ValidationError):
        ImageObservation(image_id="chart-1", confidence=0.9)
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="source:vision:0",
            evidence_kind="source_chunk",
            locator=locator,
            text="A chart supports the claim.",
            image_observations=(
                ImageObservation(image_id="not-in-locator", inspected=False),
            ),
        )
