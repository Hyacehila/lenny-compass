from __future__ import annotations

import re
from collections.abc import Sequence

from ai_pm_contracts import AnswerClaim, Evidence, derive_answer_claims

from .providers import ProviderError

_CJK = re.compile(r"[\u3400-\u9fff]")


def insufficient_evidence_answer(question: str) -> str:
    """Return a deterministic, locale-aware refusal with no invented facts."""

    if _CJK.search(question):
        return "现有证据不足，无法可靠回答这个问题。"
    return "The available evidence is insufficient to answer this question reliably."


def bind_cited_evidence(
    answer: str,
    available: Sequence[Evidence],
    *,
    error_prefix: str,
) -> tuple[tuple[Evidence, ...], tuple[AnswerClaim, ...]]:
    """Validate citation IDs and derive the persisted claim/evidence contract.

    Only Evidence referenced by an exact ``[evidence:EVIDENCE_ID]`` marker is
    returned. Candidate Evidence may be streamed for observability, but it is
    never promoted to a persisted citation merely because it was retrieved.
    """

    try:
        claims = derive_answer_claims(answer)
    except ValueError as exc:
        raise ProviderError(
            f"{error_prefix}_MISSING_CITATIONS",
            str(exc),
        ) from exc

    cited_in_order = _unique(
        tuple(evidence_id for claim in claims for evidence_id in claim.evidence_ids)
    )

    available_by_id = {item.evidence_id: item for item in available}
    unknown = [
        evidence_id
        for evidence_id in cited_in_order
        if evidence_id not in available_by_id
    ]
    if unknown:
        raise ProviderError(
            f"{error_prefix}_UNREAD_CITATION",
            "Answer cited Evidence that was not resolved in this run",
        )

    claimed_ids = {
        evidence_id for claim in claims for evidence_id in claim.evidence_ids
    }
    actual = tuple(item for item in available if item.evidence_id in claimed_ids)
    return actual, claims


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
