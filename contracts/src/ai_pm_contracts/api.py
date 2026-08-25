from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue, field_validator

from ._base import ContractModel
from .facts import DocumentChunkLocator, Evidence, NonEmpty

AnswerMode = Literal["direct", "agentic"]


class AnswerProcessPhase(ContractModel):
    phase: NonEmpty
    message: str | None = None


class AnswerProcessTrace(ContractModel):
    phases: tuple[AnswerProcessPhase, ...] = ()
    provider_counts: dict[str, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict
    )
    fused_count: Annotated[int, Field(ge=0)] | None = None
    evidence_count: Annotated[int, Field(ge=0)] = 0


class AnswerRequest(ContractModel):
    question: NonEmpty
    mode: AnswerMode = "direct"


class AnswerResponse(ContractModel):
    request_id: NonEmpty
    mode: AnswerMode
    answer: NonEmpty
    evidence: tuple[Evidence, ...] = ()


class SourceWindowRequest(ContractModel):
    locator: DocumentChunkLocator
    window: Annotated[int, Field(ge=0)] = 0


class SourceWindowResponse(ContractModel):
    locator: DocumentChunkLocator
    window: Annotated[int, Field(ge=0)]
    evidence: tuple[Evidence, ...]


class ApiError(ContractModel):
    code: NonEmpty
    message: NonEmpty
    retryable: bool = False
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ApiErrorResponse(ContractModel):
    error: ApiError


class SsePhaseEvent(ContractModel):
    type: Literal["phase"] = "phase"
    phase: NonEmpty
    message: str | None = None


class SseRetrievalSummaryEvent(ContractModel):
    type: Literal["retrieval_summary"] = "retrieval_summary"
    provider_counts: dict[str, Annotated[int, Field(ge=0)]]
    fused_count: Annotated[int, Field(ge=0)]


class SseEvidenceEvent(ContractModel):
    type: Literal["evidence"] = "evidence"
    evidence: Evidence


class SseAnswerDeltaEvent(ContractModel):
    model_config = ConfigDict(str_strip_whitespace=False)

    type: Literal["answer_delta"] = "answer_delta"
    delta: NonEmpty

    @field_validator("delta")
    @classmethod
    def validate_visible_delta(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("delta must contain a non-whitespace character")
        return value


class SseDoneEvent(ContractModel):
    type: Literal["done"] = "done"
    response: AnswerResponse


class SseErrorEvent(ContractModel):
    type: Literal["error"] = "error"
    error: ApiError


SseEvent = Annotated[
    SsePhaseEvent
    | SseRetrievalSummaryEvent
    | SseEvidenceEvent
    | SseAnswerDeltaEvent
    | SseDoneEvent
    | SseErrorEvent,
    Field(discriminator="type"),
]
