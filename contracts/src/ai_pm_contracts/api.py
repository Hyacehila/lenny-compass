from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator

from ._base import ContractModel
from .facts import DocumentChunkLocator, Evidence, NonEmpty

AnswerMode = Literal["direct", "agentic"]
AnswerKind = Literal["grounded", "insufficient_evidence"]
AnswerContractVersion = Literal["v1", "v2"]
ResponseKind = Literal[
    "grounded_answer",
    "general_guidance",
    "clarification",
    "insufficient_evidence",
]
ClaimRole = Literal["core", "supporting", "advice_rationale"]
AdviceBasis = Literal["evidence_synthesis", "general_pm_guidance"]
InsufficientReason = Literal[
    "no_evidence",
    "semantic_support_failed",
    "semantic_support_uncertain",
]
BoundedQuestion = Annotated[str, Field(min_length=1, max_length=8_000)]
BoundedAnswer = Annotated[str, Field(min_length=1, max_length=100_000)]
IdempotencyKey = Annotated[str, Field(min_length=1, max_length=200)]


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
    question: BoundedQuestion
    mode: AnswerMode = "direct"
    idempotency_key: IdempotencyKey | None = None
    coaching_id: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    reply_to_message_id: Annotated[str, Field(min_length=1, max_length=200)] | None = (
        None
    )
    skip_clarification: bool = False

    @model_validator(mode="after")
    def validate_coaching_reply(self) -> AnswerRequest:
        if self.skip_clarification and self.coaching_id is None:
            raise ValueError("skip_clarification requires coaching_id")
        if self.reply_to_message_id is not None and self.coaching_id is None:
            raise ValueError("reply_to_message_id requires coaching_id")
        if self.coaching_id is not None and self.reply_to_message_id is None:
            raise ValueError("coaching_id requires reply_to_message_id")
        return self


class AnswerClaim(ContractModel):
    """One answer claim and the exact Evidence records supporting it."""

    claim_id: NonEmpty
    text: NonEmpty
    evidence_ids: Annotated[tuple[NonEmpty, ...], Field(min_length=1)]
    role: ClaimRole = "supporting"

    @field_validator("evidence_ids")
    @classmethod
    def validate_unique_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("claim evidence_ids must be unique")
        return value


class AdviceItem(ContractModel):
    advice_id: NonEmpty
    text: NonEmpty
    basis: AdviceBasis
    rationale_claim_ids: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def validate_basis(self) -> AdviceItem:
        if len(self.rationale_claim_ids) != len(set(self.rationale_claim_ids)):
            raise ValueError("advice rationale claim ids must be unique")
        if self.basis == "evidence_synthesis" and not self.rationale_claim_ids:
            raise ValueError("evidence_synthesis advice requires rationale claims")
        if self.basis == "general_pm_guidance" and self.rationale_claim_ids:
            raise ValueError("general_pm_guidance cannot claim evidence rationale")
        return self


class UserPremise(ContractModel):
    premise_id: NonEmpty
    text: NonEmpty
    source_message_id: NonEmpty


class Clarification(ContractModel):
    coaching_id: NonEmpty
    round: Annotated[int, Field(ge=1, le=3)]
    prompt: NonEmpty
    suggestions: Annotated[tuple[NonEmpty, ...], Field(max_length=4)] = ()
    can_skip: Literal[True] = True

    @field_validator("suggestions")
    @classmethod
    def validate_unique_suggestions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("clarification suggestions must be unique")
        return value


class AnswerVerification(ContractModel):
    status: Literal["passed"] = "passed"
    scope: Literal["claim_evidence_support"] = "claim_evidence_support"
    policy_version: NonEmpty
    checked_claim_count: Annotated[int, Field(ge=1)]
    omitted_claim_count: Annotated[int, Field(ge=0)] = 0
    calibration_level: Literal["uncalibrated", "minimal"] = "uncalibrated"


_CITATION = re.compile(r"\[evidence:([^\]\r\n]+)\]")
_STRUCTURAL_LINE = re.compile(r"^(?:\s*```|\s*(?:-{3,}|\*{3,}|_{3,})\s*$)")


def derive_answer_claims(answer: str) -> tuple[AnswerClaim, ...]:
    """Derive the canonical per-line claim/citation representation."""

    claims: list[AnswerClaim] = []
    for raw_line in answer.splitlines() or [answer]:
        visible = raw_line.strip()
        if not visible or _STRUCTURAL_LINE.match(visible):
            continue
        evidence_ids = tuple(
            dict.fromkeys(
                value.strip() for value in _CITATION.findall(raw_line) if value.strip()
            )
        )
        if not evidence_ids:
            raise ValueError("every visible answer line must cite Evidence")
        text = _CITATION.sub("", raw_line).strip()
        if not text:
            raise ValueError("citation markers must accompany visible claim text")
        claims.append(
            AnswerClaim(
                claim_id=f"claim-{len(claims) + 1}",
                text=text,
                evidence_ids=evidence_ids,
            )
        )
    if not claims:
        raise ValueError("grounded answer contains no cited claims")
    return tuple(claims)


class AnswerResponse(ContractModel):
    request_id: NonEmpty
    assistant_message_id: NonEmpty | None = None
    mode: AnswerMode
    contract_version: AnswerContractVersion = "v1"
    response_kind: ResponseKind | None = None
    answer_kind: AnswerKind | None = "grounded"
    answer: BoundedAnswer
    evidence: tuple[Evidence, ...] = ()
    claims: tuple[AnswerClaim, ...] = ()
    advice: tuple[AdviceItem, ...] = ()
    user_premises: tuple[UserPremise, ...] = ()
    clarification: Clarification | None = None
    verification: AnswerVerification | None = None
    insufficient_reason: InsufficientReason | None = None

    @model_validator(mode="after")
    def validate_grounding(self) -> AnswerResponse:
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("answer evidence IDs must be unique")

        if self.contract_version == "v2":
            return self._validate_v2(evidence_ids)

        if self.response_kind is not None:
            raise ValueError("v1 answers cannot define response_kind")
        if self.advice or self.user_premises or self.clarification or self.verification:
            raise ValueError("v1 answers cannot define Stage 2 payload fields")
        if self.insufficient_reason is not None:
            raise ValueError("v1 answers cannot define insufficient_reason")

        if self.answer_kind == "insufficient_evidence":
            if self.evidence or self.claims:
                raise ValueError(
                    "insufficient_evidence answers cannot include evidence or claims"
                )
            return self

        if not self.evidence or not self.claims:
            raise ValueError("grounded answers require evidence and claims")
        if self.claims != derive_answer_claims(self.answer):
            raise ValueError("answer claims must exactly match cited answer lines")
        claimed_ids = {
            evidence_id for claim in self.claims for evidence_id in claim.evidence_ids
        }
        if claimed_ids != set(evidence_ids):
            raise ValueError(
                "answer evidence must contain exactly the Evidence cited by claims"
            )
        return self

    def _validate_v2(self, evidence_ids: tuple[str, ...]) -> AnswerResponse:
        if self.response_kind is None:
            raise ValueError("v2 answers require response_kind")

        claim_ids = tuple(item.claim_id for item in self.claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("answer claim IDs must be unique")
        advice_ids = tuple(item.advice_id for item in self.advice)
        if len(advice_ids) != len(set(advice_ids)):
            raise ValueError("answer advice IDs must be unique")
        premise_ids = tuple(item.premise_id for item in self.user_premises)
        if len(premise_ids) != len(set(premise_ids)):
            raise ValueError("answer premise IDs must be unique")

        if self.response_kind == "grounded_answer":
            if self.answer_kind != "grounded":
                raise ValueError("grounded_answer requires answer_kind=grounded")
            if not self.evidence or not self.claims:
                raise ValueError("grounded_answer requires evidence and claims")
            if not any(item.role == "core" for item in self.claims):
                raise ValueError("grounded_answer requires at least one core claim")
            if self.clarification is not None or self.insufficient_reason is not None:
                raise ValueError("grounded_answer cannot include refusal or clarification")
            if self.verification is None:
                raise ValueError("grounded_answer requires semantic verification")
            if self.verification.checked_claim_count != len(self.claims):
                raise ValueError("verification count must match published claims")
            claimed_ids = {
                evidence_id
                for claim in self.claims
                for evidence_id in claim.evidence_ids
            }
            if claimed_ids != set(evidence_ids):
                raise ValueError(
                    "grounded answer evidence must exactly match published claims"
                )
            available_claims = set(claim_ids)
            for item in self.advice:
                if any(
                    claim_id not in available_claims
                    for claim_id in item.rationale_claim_ids
                ):
                    raise ValueError("advice references an unknown rationale claim")
            return self

        if self.response_kind == "general_guidance":
            if self.answer_kind is not None:
                raise ValueError("general_guidance cannot define answer_kind")
            if self.evidence or self.claims or self.verification:
                raise ValueError("general_guidance cannot claim local evidence support")
            if not self.advice or any(
                item.basis != "general_pm_guidance" for item in self.advice
            ):
                raise ValueError("general_guidance requires general PM advice")
            if self.clarification is not None:
                raise ValueError("general_guidance cannot include clarification")
            if self.insufficient_reason != "no_evidence":
                raise ValueError("general_guidance must disclose missing local evidence")
            return self

        if self.response_kind == "clarification":
            if self.answer_kind is not None or self.clarification is None:
                raise ValueError("clarification response has invalid answer fields")
            if (
                self.evidence
                or self.claims
                or self.advice
                or self.user_premises
                or self.verification
                or self.insufficient_reason is not None
            ):
                raise ValueError("clarification response cannot include answer payload")
            return self

        if self.answer_kind != "insufficient_evidence":
            raise ValueError(
                "insufficient_evidence response requires matching answer_kind"
            )
        if (
            self.evidence
            or self.claims
            or self.advice
            or self.user_premises
            or self.clarification
            or self.verification
        ):
            raise ValueError("insufficient_evidence cannot include answer payload")
        if self.insufficient_reason is None:
            raise ValueError("insufficient_evidence requires a reason")
        return self


class SourceWindowRequest(ContractModel):
    locator: DocumentChunkLocator
    window: Annotated[int, Field(ge=0, le=8)] = 0


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
