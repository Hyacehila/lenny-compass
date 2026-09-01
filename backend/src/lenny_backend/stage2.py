from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Protocol

from ai_pm_contracts import (
    AdviceItem,
    AnswerClaim,
    AnswerResponse,
    AnswerVerification,
    Clarification,
    Evidence,
    QueryRewrite,
    UserPremise,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .grounding import insufficient_evidence_answer
from .providers import ProviderError

STAGE2_POLICY_VERSION = "claim-support-minimal-v1"
MAX_COACHING_ROUNDS = 3
_CJK = re.compile(r"[\u3400-\u9fff]")


class Stage2Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class QuestionPlan(Stage2Model):
    intent: Literal["knowledge", "comparison", "scenario_advice", "follow_up"]
    standalone_question: Annotated[str, Field(min_length=1, max_length=8_000)]
    semantic_query_en: Annotated[str, Field(min_length=1, max_length=2_000)]
    lexical_terms_en: tuple[Annotated[str, Field(min_length=1, max_length=200)], ...] = ()
    lexical_phrases_en: tuple[Annotated[str, Field(min_length=1, max_length=500)], ...] = ()
    source_scope: Literal["broad", "single_document"] = "broad"
    needs_clarification: bool = False
    clarification_prompt: Annotated[str, Field(max_length=1_000)] | None = None
    clarification_suggestions: tuple[
        Annotated[str, Field(min_length=1, max_length=300)], ...
    ] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def clarification_is_consistent(self) -> QuestionPlan:
        if self.needs_clarification and not self.clarification_prompt:
            raise ValueError("clarification plan requires a prompt")
        if not self.needs_clarification and (
            self.clarification_prompt or self.clarification_suggestions
        ):
            raise ValueError("non-clarification plan cannot include clarification copy")
        if len(self.clarification_suggestions) != len(
            set(self.clarification_suggestions)
        ):
            raise ValueError("clarification suggestions must be unique")
        return self

    def as_rewrite(self, original_question: str) -> QueryRewrite:
        return QueryRewrite(
            original_query=original_question,
            semantic_query_en=self.semantic_query_en,
            lexical_terms_en=self.lexical_terms_en[:8],
            lexical_phrases_en=self.lexical_phrases_en[:8],
            used_fallback=False,
        )


class DraftClaim(Stage2Model):
    claim_id: Annotated[str, Field(min_length=1, max_length=160)]
    text: Annotated[str, Field(min_length=1, max_length=4_000)]
    role: Literal["core", "supporting", "advice_rationale"]
    evidence_ids: tuple[
        Annotated[str, Field(min_length=1, max_length=1_024)], ...
    ] = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("draft claim evidence ids must be unique")
        return value


class DraftAdvice(Stage2Model):
    advice_id: Annotated[str, Field(min_length=1, max_length=160)]
    text: Annotated[str, Field(min_length=1, max_length=4_000)]
    basis: Literal["evidence_synthesis", "general_pm_guidance"]
    rationale_claim_ids: tuple[
        Annotated[str, Field(min_length=1, max_length=160)], ...
    ] = ()

    @model_validator(mode="after")
    def basis_is_consistent(self) -> DraftAdvice:
        if self.basis == "evidence_synthesis" and not self.rationale_claim_ids:
            raise ValueError("evidence_synthesis requires rationale claims")
        if self.basis == "general_pm_guidance" and self.rationale_claim_ids:
            raise ValueError("general guidance cannot claim source rationale")
        if len(self.rationale_claim_ids) != len(set(self.rationale_claim_ids)):
            raise ValueError("rationale claim ids must be unique")
        return self


class DraftPremise(Stage2Model):
    premise_id: Annotated[str, Field(min_length=1, max_length=160)]
    text: Annotated[str, Field(min_length=1, max_length=2_000)]
    source_message_id: Annotated[str, Field(min_length=1, max_length=200)]


class AnswerDraft(Stage2Model):
    response_kind: Literal[
        "grounded_answer", "general_guidance", "insufficient_evidence"
    ]
    claims: tuple[DraftClaim, ...] = ()
    advice: tuple[DraftAdvice, ...] = ()
    user_premises: tuple[DraftPremise, ...] = ()

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> AnswerDraft:
        for label, values in (
            ("claim", tuple(item.claim_id for item in self.claims)),
            ("advice", tuple(item.advice_id for item in self.advice)),
            ("premise", tuple(item.premise_id for item in self.user_premises)),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} ids")
        if self.response_kind == "grounded_answer":
            if not self.claims or not any(item.role == "core" for item in self.claims):
                raise ValueError("grounded draft requires a core claim")
        elif self.response_kind == "general_guidance":
            if self.claims or not self.advice or any(
                item.basis != "general_pm_guidance" for item in self.advice
            ):
                raise ValueError("general guidance draft has invalid content")
        elif self.claims or self.advice or self.user_premises:
            raise ValueError("insufficient draft cannot contain answer content")
        return self


class ClaimVerdict(Stage2Model):
    claim_id: Annotated[str, Field(min_length=1, max_length=160)]
    verdict: Literal["supported", "insufficient_context", "contradicted"]


class VerificationBatch(Stage2Model):
    verdicts: tuple[ClaimVerdict, ...] = ()
    general_guidance_safe: bool = True

    @model_validator(mode="after")
    def verdict_ids_are_unique(self) -> VerificationBatch:
        ids = tuple(item.claim_id for item in self.verdicts)
        if len(ids) != len(set(ids)):
            raise ValueError("verification verdict ids must be unique")
        return self


class Stage2Provider(Protocol):
    async def plan_question(
        self,
        *,
        question: str,
        conversation_context: str,
        completed_clarification_rounds: int,
        skip_clarification: bool,
    ) -> QuestionPlan: ...

    async def draft_stage2_answer(
        self,
        *,
        question: str,
        conversation_context: str,
        evidence: Sequence[Evidence],
        message_ids: Sequence[str],
    ) -> AnswerDraft: ...

    async def verify_stage2_answer(
        self,
        *,
        draft: AnswerDraft,
        evidence: Sequence[Evidence],
        conversation_context: str,
    ) -> VerificationBatch: ...

    async def repair_stage2_answer(
        self,
        *,
        question: str,
        draft: AnswerDraft,
        verification: VerificationBatch,
        evidence: Sequence[Evidence],
        conversation_context: str,
        message_ids: Sequence[str],
    ) -> AnswerDraft: ...


def supports_stage2(provider: Any) -> bool:
    methods_ready = all(
        callable(getattr(provider, name, None))
        for name in (
            "plan_question",
            "draft_stage2_answer",
            "verify_stage2_answer",
            "repair_stage2_answer",
        )
    )
    configured_prompts = tuple(
        getattr(provider, name)
        for name in (
            "plan_prompt",
            "stage2_answer_prompt",
            "verify_prompt",
            "repair_prompt",
        )
        if hasattr(provider, name)
    )
    return methods_ready and (
        not configured_prompts or all(bool(value) for value in configured_prompts)
    )


def coaching_state(
    history: Sequence[Any], *, coaching_id: str | None, reply_to_message_id: str | None
) -> tuple[int, str | None]:
    if coaching_id is None:
        if reply_to_message_id is not None:
            raise ProviderError(
                "INVALID_COACHING_CONTEXT",
                "Clarification reply is missing its coaching identifier",
            )
        return 0, None
    if not history:
        raise ProviderError(
            "INVALID_COACHING_CONTEXT", "Clarification context is no longer available"
        )
    latest = next(
        (item for item in reversed(history) if getattr(item, "role", None) == "assistant"),
        None,
    )
    if latest is None:
        raise ProviderError(
            "INVALID_COACHING_CONTEXT", "Clarification context is no longer available"
        )
    payload = getattr(latest, "response_payload", {}) or {}
    clarification = payload.get("clarification") if isinstance(payload, dict) else None
    if (
        getattr(latest, "id", None) != reply_to_message_id
        or not isinstance(clarification, dict)
        or clarification.get("coaching_id") != coaching_id
    ):
        raise ProviderError(
            "INVALID_COACHING_CONTEXT", "Clarification reply does not match this conversation"
        )
    round_value = clarification.get("round")
    if not isinstance(round_value, int) or not 1 <= round_value <= MAX_COACHING_ROUNDS:
        raise ProviderError(
            "INVALID_COACHING_CONTEXT", "Stored clarification round is invalid"
        )
    return round_value, coaching_id


def make_clarification_response(
    *,
    request_id: str,
    mode: Literal["direct", "agentic"],
    plan: QuestionPlan,
    completed_rounds: int,
    coaching_id: str | None,
) -> AnswerResponse:
    next_round = completed_rounds + 1
    if next_round > MAX_COACHING_ROUNDS or not plan.clarification_prompt:
        raise ValueError("clarification exceeds the allowed coaching rounds")
    clarification = Clarification(
        coaching_id=coaching_id or str(uuid.uuid4()),
        round=next_round,
        prompt=plan.clarification_prompt,
        suggestions=plan.clarification_suggestions,
    )
    return AnswerResponse(
        request_id=request_id,
        mode=mode,
        contract_version="v2",
        response_kind="clarification",
        answer_kind=None,
        answer=clarification.prompt,
        clarification=clarification,
    )


async def finalize_stage2_answer(
    *,
    provider: Stage2Provider,
    mode: Literal["direct", "agentic"],
    request_id: str,
    question: str,
    conversation_context: str,
    evidence: Sequence[Evidence],
    history: Sequence[Any],
    initial_draft: AnswerDraft | None = None,
) -> AnswerResponse:
    message_ids = tuple(str(item.id) for item in history)
    draft = initial_draft or await provider.draft_stage2_answer(
        question=question,
        conversation_context=conversation_context,
        evidence=evidence,
        message_ids=message_ids,
    )
    draft, structurally_omitted = _sanitize_draft_references(
        draft, evidence, message_ids
    )

    try:
        verification = await provider.verify_stage2_answer(
            draft=draft,
            evidence=evidence,
            conversation_context=conversation_context,
        )
    except ProviderError:
        return _insufficient_response(
            request_id=request_id,
            mode=mode,
            question=question,
            reason="semantic_support_uncertain",
        )
    _validate_verification(draft, verification)
    rejected_claim_ids = set(structurally_omitted)
    rejected_claim_ids.update(_failed_claim_ids(verification))
    if rejected_claim_ids or not verification.general_guidance_safe:
        try:
            draft = await provider.repair_stage2_answer(
                question=question,
                draft=draft,
                verification=verification,
                evidence=evidence,
                conversation_context=conversation_context,
                message_ids=message_ids,
            )
        except ProviderError:
            return _insufficient_response(
                request_id=request_id,
                mode=mode,
                question=question,
                reason="semantic_support_uncertain",
            )
        draft, repaired_structural_omissions = _sanitize_draft_references(
            draft, evidence, message_ids
        )
        rejected_claim_ids.update(repaired_structural_omissions)
        try:
            verification = await provider.verify_stage2_answer(
                draft=draft,
                evidence=evidence,
                conversation_context=conversation_context,
            )
        except ProviderError:
            return _insufficient_response(
                request_id=request_id,
                mode=mode,
                question=question,
                reason="semantic_support_uncertain",
            )
        _validate_verification(draft, verification)

    failed = _failed_claim_ids(verification)
    rejected_claim_ids.update(failed)
    failed_count = len(rejected_claim_ids)
    safe_general = verification.general_guidance_safe
    kept_claims = tuple(item for item in draft.claims if item.claim_id not in failed)
    core_failed = draft.response_kind == "grounded_answer" and not any(
        item.role == "core" and item.claim_id not in failed for item in draft.claims
    )
    kept_claim_ids = {item.claim_id for item in kept_claims}
    kept_advice = tuple(
        item
        for item in draft.advice
        if (
            (item.basis == "general_pm_guidance" and safe_general)
            or (
                item.basis == "evidence_synthesis"
                and set(item.rationale_claim_ids) <= kept_claim_ids
            )
        )
    )

    if draft.response_kind == "general_guidance" or core_failed:
        general = tuple(
            item for item in kept_advice if item.basis == "general_pm_guidance"
        )
        if general:
            advice = tuple(_advice(item) for item in general)
            return AnswerResponse(
                request_id=request_id,
                mode=mode,
                contract_version="v2",
                response_kind="general_guidance",
                answer_kind=None,
                answer=render_stage2_answer(
                    question=question,
                    claims=(),
                    advice=advice,
                    user_premises=(),
                    omitted_claim_count=failed_count,
                    general_guidance_only=True,
                ),
                advice=advice,
                insufficient_reason="no_evidence",
            )
        return _insufficient_response(
            request_id=request_id,
            mode=mode,
            question=question,
            reason=(
                "semantic_support_failed" if failed else "no_evidence"
            ),
        )

    if draft.response_kind == "insufficient_evidence" or not kept_claims:
        return _insufficient_response(
            request_id=request_id,
            mode=mode,
            question=question,
            reason="no_evidence" if not draft.claims else "semantic_support_failed",
        )

    claims = tuple(_claim(item) for item in kept_claims)
    advice = tuple(_advice(item) for item in kept_advice)
    premises = tuple(_premise(item) for item in draft.user_premises)
    used_evidence_ids = {
        evidence_id for claim in claims for evidence_id in claim.evidence_ids
    }
    used_evidence = tuple(
        item for item in evidence if item.evidence_id in used_evidence_ids
    )
    return AnswerResponse(
        request_id=request_id,
        mode=mode,
        contract_version="v2",
        response_kind="grounded_answer",
        answer_kind="grounded",
        answer=render_stage2_answer(
            question=question,
            claims=claims,
            advice=advice,
            user_premises=premises,
            omitted_claim_count=failed_count,
            general_guidance_only=False,
        ),
        evidence=used_evidence,
        claims=claims,
        advice=advice,
        user_premises=premises,
        verification=AnswerVerification(
            policy_version=STAGE2_POLICY_VERSION,
            checked_claim_count=len(claims),
            omitted_claim_count=failed_count,
        ),
    )


def render_stage2_answer(
    *,
    question: str,
    claims: Sequence[AnswerClaim],
    advice: Sequence[AdviceItem],
    user_premises: Sequence[UserPremise],
    omitted_claim_count: int,
    general_guidance_only: bool,
) -> str:
    zh = bool(_CJK.search(question))
    lines: list[str] = []
    if general_guidance_only:
        lines.append(
            "本地资料没有提供足够证据。下面是模型给出的通用 PM 建议，不代表资料库结论。"
            if zh
            else "The local corpus did not provide enough evidence. The following is general PM guidance from the model, not a corpus finding."
        )
    elif claims:
        lines.append("## 结论" if zh else "## Findings")
        for item in claims:
            markers = " ".join(f"[evidence:{value}]" for value in item.evidence_ids)
            lines.append(f"- {item.text} {markers}")
    if user_premises:
        lines.append("\n## 基于你提供的信息" if zh else "\n## Based on your context")
        lines.extend(f"- {item.text}" for item in user_premises)
    if advice:
        lines.append("\n## 建议" if zh else "\n## Recommendations")
        claim_by_id = {item.claim_id: item for item in claims}
        for item in advice:
            if item.basis == "general_pm_guidance":
                suffix = "（模型通用建议）" if zh else " (general model guidance)"
                lines.append(f"- {item.text}{suffix}")
                continue
            evidence_ids = tuple(
                dict.fromkeys(
                    evidence_id
                    for claim_id in item.rationale_claim_ids
                    for evidence_id in claim_by_id[claim_id].evidence_ids
                )
            )
            markers = " ".join(f"[evidence:{value}]" for value in evidence_ids)
            lines.append(f"- {item.text} {markers}")
    if omitted_claim_count:
        lines.append(
            f"\n已省略 {omitted_claim_count} 条未通过证据核验的内容。"
            if zh
            else f"\n{omitted_claim_count} unsupported item(s) were omitted."
        )
    return "\n".join(lines).strip()


def _claim(item: DraftClaim) -> AnswerClaim:
    return AnswerClaim(
        claim_id=item.claim_id,
        text=item.text,
        role=item.role,
        evidence_ids=item.evidence_ids,
    )


def _advice(item: DraftAdvice) -> AdviceItem:
    return AdviceItem(
        advice_id=item.advice_id,
        text=item.text,
        basis=item.basis,
        rationale_claim_ids=item.rationale_claim_ids,
    )


def _premise(item: DraftPremise) -> UserPremise:
    return UserPremise(
        premise_id=item.premise_id,
        text=item.text,
        source_message_id=item.source_message_id,
    )


def _validate_draft_references(
    draft: AnswerDraft,
    evidence: Sequence[Evidence],
    message_ids: Sequence[str],
) -> None:
    available_evidence = {item.evidence_id for item in evidence}
    available_messages = set(message_ids)
    claim_by_id = {item.claim_id: item for item in draft.claims}
    unknown_evidence = {
        value
        for item in draft.claims
        for value in item.evidence_ids
        if value not in available_evidence
    }
    if unknown_evidence:
        raise ProviderError(
            "STAGE2_UNKNOWN_EVIDENCE",
            "Stage 2 draft cited evidence that was not resolved in this run",
        )
    if any(
        item.source_message_id not in available_messages for item in draft.user_premises
    ):
        raise ProviderError(
            "STAGE2_UNKNOWN_USER_PREMISE",
            "Stage 2 draft attributed a premise to an unknown message",
        )
    for item in draft.advice:
        for claim_id in item.rationale_claim_ids:
            if claim_by_id.get(claim_id) is None:
                raise ProviderError(
                    "STAGE2_INVALID_ADVICE_RATIONALE",
                    "Stage 2 advice referenced an invalid rationale claim",
                )


def _sanitize_draft_references(
    draft: AnswerDraft,
    evidence: Sequence[Evidence],
    message_ids: Sequence[str],
) -> tuple[AnswerDraft, set[str]]:
    """Drop invented references before semantic verification, preserving safe content."""

    available_evidence = {item.evidence_id for item in evidence}
    available_messages = set(message_ids)
    claims = tuple(
        item
        for item in draft.claims
        if set(item.evidence_ids) <= available_evidence
    )
    omitted = {item.claim_id for item in draft.claims} - {
        item.claim_id for item in claims
    }
    claim_ids = {item.claim_id for item in claims}
    advice: list[DraftAdvice] = []
    for item in draft.advice:
        if item.basis == "general_pm_guidance":
            advice.append(item)
            continue
        rationale_ids = tuple(
            claim_id for claim_id in item.rationale_claim_ids if claim_id in claim_ids
        )
        if rationale_ids:
            advice.append(item.model_copy(update={"rationale_claim_ids": rationale_ids}))
    premises = tuple(
        item
        for item in draft.user_premises
        if item.source_message_id in available_messages
    )

    if draft.response_kind == "grounded_answer" and not any(
        item.role == "core" for item in claims
    ):
        general = tuple(
            item for item in advice if item.basis == "general_pm_guidance"
        )
        if general:
            return AnswerDraft(response_kind="general_guidance", advice=general), omitted
        return AnswerDraft(response_kind="insufficient_evidence"), omitted

    sanitized = AnswerDraft(
        response_kind=draft.response_kind,
        claims=claims,
        advice=tuple(advice),
        user_premises=premises,
    )
    _validate_draft_references(sanitized, evidence, message_ids)
    return sanitized, omitted


def _validate_verification(
    draft: AnswerDraft, verification: VerificationBatch
) -> None:
    expected = {item.claim_id for item in draft.claims}
    actual = {item.claim_id for item in verification.verdicts}
    if expected != actual:
        raise ProviderError(
            "SEMANTIC_VERIFICATION_INVALID_RESPONSE",
            "Semantic verifier did not return exactly one verdict per claim",
            retryable=True,
        )


def _failed_claim_ids(verification: VerificationBatch) -> set[str]:
    return {
        item.claim_id
        for item in verification.verdicts
        if item.verdict != "supported"
    }


def _insufficient_response(
    *,
    request_id: str,
    mode: Literal["direct", "agentic"],
    question: str,
    reason: Literal[
        "no_evidence", "semantic_support_failed", "semantic_support_uncertain"
    ],
) -> AnswerResponse:
    return AnswerResponse(
        request_id=request_id,
        mode=mode,
        contract_version="v2",
        response_kind="insufficient_evidence",
        answer_kind="insufficient_evidence",
        answer=insufficient_evidence_answer(question),
        insufficient_reason=reason,
    )
