from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ._base import ContractModel
from .facts import EvidenceLocator, NonEmpty, Sha256


class RetrievalProvider(StrEnum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    LLM_WIKI = "llm_wiki"
    HYBRID = "hybrid"


class QueryRewrite(ContractModel):
    """Validated output of the corpus-specific query rewriting stage."""

    original_query: NonEmpty
    semantic_query_en: NonEmpty
    lexical_terms_en: Annotated[tuple[NonEmpty, ...], Field(max_length=8)] = ()
    lexical_phrases_en: Annotated[tuple[NonEmpty, ...], Field(max_length=8)] = ()
    used_fallback: bool = False


class RetrievalIndexStatus(ContractModel):
    """Runtime identity/readiness stored inside one standalone SQLite index."""

    index_kind: Literal["lexical", "semantic"]
    schema_version: NonEmpty
    build_id: NonEmpty
    facts_manifest_sha256: Sha256
    status: Literal["ready", "building", "test_only"]
    embedding_model: NonEmpty | None = None
    embedding_dimensions: Literal[2048] | None = None

    @model_validator(mode="after")
    def validate_index_specific_fields(self) -> RetrievalIndexStatus:
        if self.index_kind == "lexical":
            if self.status == "test_only":
                raise ValueError("lexical index cannot have test_only status")
            if (
                self.embedding_model is not None
                or self.embedding_dimensions is not None
            ):
                raise ValueError("lexical index cannot declare embedding fields")
        elif self.embedding_model is None or self.embedding_dimensions != 2048:
            raise ValueError("semantic index requires model and 2048 dimensions")
        return self


class RetrievalHit(ContractModel):
    """Ranked locator only; deliberately contains no source excerpt."""

    provider: Literal[
        RetrievalProvider.KEYWORD,
        RetrievalProvider.VECTOR,
        RetrievalProvider.LLM_WIKI,
    ]
    rank: Annotated[int, Field(ge=1)]
    native_score: float
    locator: EvidenceLocator
    retrieval_segment_ordinal: Annotated[int, Field(ge=0)] | None = None


class FusedRetrievalHit(ContractModel):
    """RRF result retaining component ranks for an auditable trace."""

    provider: Literal[RetrievalProvider.HYBRID] = RetrievalProvider.HYBRID
    rank: Annotated[int, Field(ge=1)]
    rrf_score: Annotated[float, Field(gt=0)]
    locator: EvidenceLocator
    component_ranks: dict[str, Annotated[int, Field(ge=1)]]
    component_scores: dict[str, float]


class SearchRequest(ContractModel):
    question: NonEmpty
    providers: tuple[RetrievalProvider, ...] = (
        RetrievalProvider.KEYWORD,
        RetrievalProvider.VECTOR,
    )
    limit: Annotated[int, Field(ge=1, le=50)] = 50


class SearchResponse(ContractModel):
    question: NonEmpty
    hits: tuple[RetrievalHit | FusedRetrievalHit, ...]
