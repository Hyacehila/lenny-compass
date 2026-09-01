from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

from .io import EvalInputError
from .models import (
    RESULT_SCHEMA_VERSION,
    BehaviorMetrics,
    CaseObservation,
    CaseScore,
    CaseSuite,
    DistributionMetrics,
    EvaluationCase,
    EvaluationResult,
    GroundingMetrics,
    ObservationSet,
    OperationsMetrics,
    RetrievalMetrics,
)


def _score(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 12)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(fmean(values), 12)


def _distribution(values: list[float]) -> DistributionMetrics:
    if not values:
        return DistributionMetrics(count=0, mean=None, p50=None, p95=None, max=None)
    ordered = sorted(values)

    def nearest_rank(percentile: float) -> float:
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[index]

    return DistributionMetrics(
        count=len(values),
        mean=_mean(values),
        p50=nearest_rank(0.50),
        p95=nearest_rank(0.95),
        max=ordered[-1],
    )


def _retrieval_scores(
    case: EvaluationCase, observation: CaseObservation, ks: tuple[int, ...]
) -> tuple[dict[str, float] | None, float | None]:
    relevant = set(case.expectation.relevant_evidence_ids)
    if not relevant:
        return None, None
    ranked = observation.retrieved_evidence_ids
    recalls = {
        str(k): round(len(relevant.intersection(ranked[:k])) / len(relevant), 12)
        for k in ks
    }
    first_rank = next(
        (
            rank
            for rank, evidence_id in enumerate(ranked, start=1)
            if evidence_id in relevant
        ),
        None,
    )
    mrr = 0.0 if first_rank is None else round(1 / first_rank, 12)
    return recalls, mrr


@dataclass(frozen=True)
class GroundingCounts:
    citations: int
    valid_citations: int
    claims: int
    covered_claims: int
    expected_claims: int
    addressed_expected_claims: int
    grounded_expected_claims: int


def _grounding_counts(
    case: EvaluationCase, observation: CaseObservation
) -> GroundingCounts:
    if observation.answer_kind != "answer":
        return GroundingCounts(0, 0, 0, 0, len(case.expectation.expected_claims), 0, 0)

    available = set(observation.grounded_evidence_ids)
    citations = [
        citation_id
        for claim in observation.claims
        for citation_id in claim.citation_ids
    ]
    valid_citations = sum(citation_id in available for citation_id in citations)
    covered_claims = sum(
        bool(set(claim.citation_ids) & available) for claim in observation.claims
    )

    observed_by_expectation = {
        claim.expectation_id: claim
        for claim in observation.claims
        if claim.expectation_id is not None
    }
    addressed = 0
    grounded = 0
    for expected in case.expectation.expected_claims:
        observed = observed_by_expectation.get(expected.claim_id)
        if observed is None:
            continue
        addressed += 1
        valid_for_claim = (
            set(observed.citation_ids)
            & available
            & set(expected.acceptable_evidence_ids)
        )
        grounded += bool(valid_for_claim)
    return GroundingCounts(
        citations=len(citations),
        valid_citations=valid_citations,
        claims=len(observation.claims),
        covered_claims=covered_claims,
        expected_claims=len(case.expectation.expected_claims),
        addressed_expected_claims=addressed,
        grounded_expected_claims=grounded,
    )


def _trace_compliant(case: EvaluationCase, observation: CaseObservation) -> bool | None:
    expected = case.expectation.trace
    if expected is None:
        return None
    actual = observation.trace
    if actual is None:
        return False
    successful_tools = {event.tool for event in actual.events if event.status == "ok"}
    attempted_tools = {event.tool for event in actual.events}
    return all(
        (
            set(expected.required_tools) <= successful_tools,
            not (set(expected.forbidden_tools) & attempted_tools),
            expected.max_steps is None or len(actual.events) <= expected.max_steps,
            not expected.require_completed or actual.completed,
        )
    )


def _budget_pass(actual: float | None, maximum: float | None) -> bool | None:
    if maximum is None:
        return None
    if actual is None:
        return False
    return actual <= maximum


def _append_issue(issues: list[str], condition: bool, code: str) -> None:
    if condition:
        issues.append(code)


def evaluate(
    suite: CaseSuite, observations: ObservationSet, ks: tuple[int, ...] = (1, 3, 5, 10)
) -> EvaluationResult:
    if suite.suite_id != observations.suite_id:
        raise EvalInputError(
            "SUITE_ID_MISMATCH",
            f"case suite {suite.suite_id!r} does not match observations {observations.suite_id!r}",
        )
    if not ks or any(k <= 0 for k in ks) or len(set(ks)) != len(ks):
        raise EvalInputError("INVALID_KS", "ks must be unique positive integers")
    ks = tuple(sorted(ks))

    case_ids = tuple(case.case_id for case in suite.cases)
    observation_ids = tuple(case.case_id for case in observations.cases)
    missing = sorted(set(case_ids) - set(observation_ids))
    extra = sorted(set(observation_ids) - set(case_ids))
    if missing or extra:
        raise EvalInputError(
            "CASE_SET_MISMATCH",
            f"observation case set mismatch; missing={missing}, extra={extra}",
        )
    observed_by_id = {case.case_id: case for case in observations.cases}

    per_case: list[CaseScore] = []
    retrieval_recalls: dict[int, list[float]] = {k: [] for k in ks}
    retrieval_mrrs: list[float] = []
    grounding_totals = GroundingCounts(0, 0, 0, 0, 0, 0, 0)
    eligible_answer_cases = 0
    citation_validity_scores: list[float] = []
    citation_coverage_scores: list[float] = []
    completed_cases = 0
    answer_kind_eligible = 0
    answer_kind_correct_count = 0
    refusal_eligible = 0
    refusal_correct_count = 0
    trace_eligible = 0
    trace_compliant_count = 0
    latencies: list[float] = []
    costs: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0
    budget_counts: dict[str, list[bool]] = {
        "latency": [],
        "cost": [],
        "input": [],
        "output": [],
    }

    for case in suite.cases:
        observation = observed_by_id[case.case_id]
        issues: list[str] = []
        completed = observation.outcome == "completed"
        completed_cases += completed
        _append_issue(issues, not completed, f"OUTCOME_{observation.outcome.upper()}")

        recalls, mrr = _retrieval_scores(case, observation, ks)
        if recalls is not None:
            for k in ks:
                retrieval_recalls[k].append(recalls[str(k)])
            assert mrr is not None
            retrieval_mrrs.append(mrr)
            _append_issue(issues, recalls[str(max(ks))] < 1, "RELEVANT_EVIDENCE_MISSED")

        counts = _grounding_counts(case, observation)
        grounding_totals = GroundingCounts(
            grounding_totals.citations + counts.citations,
            grounding_totals.valid_citations + counts.valid_citations,
            grounding_totals.claims + counts.claims,
            grounding_totals.covered_claims + counts.covered_claims,
            grounding_totals.expected_claims + counts.expected_claims,
            grounding_totals.addressed_expected_claims
            + counts.addressed_expected_claims,
            grounding_totals.grounded_expected_claims + counts.grounded_expected_claims,
        )
        if observation.answer_kind == "answer":
            eligible_answer_cases += 1
        citation_validity = _score(counts.valid_citations, counts.citations)
        if observation.answer_kind == "answer" and counts.citations == 0:
            citation_validity = 0.0
        citation_coverage = _score(counts.covered_claims, counts.claims)
        if observation.answer_kind == "answer" and counts.claims == 0:
            citation_coverage = 0.0
        if observation.answer_kind == "answer":
            assert citation_validity is not None and citation_coverage is not None
            citation_validity_scores.append(citation_validity)
            citation_coverage_scores.append(citation_coverage)
        expected_claim_coverage = _score(
            counts.addressed_expected_claims, counts.expected_claims
        )
        grounding_score = _score(
            counts.grounded_expected_claims, counts.addressed_expected_claims
        )
        if counts.expected_claims and not counts.addressed_expected_claims:
            grounding_score = 0.0
        _append_issue(
            issues,
            observation.answer_kind == "answer" and citation_validity < 1,
            "INVALID_OR_MISSING_CITATION",
        )
        _append_issue(
            issues,
            observation.answer_kind == "answer" and citation_coverage < 1,
            "UNCITED_CLAIM",
        )
        _append_issue(
            issues,
            grounding_score is not None and grounding_score < 1,
            "EXPECTED_CLAIM_NOT_GROUNDED",
        )
        expected_claim_ids = {
            expected.claim_id for expected in case.expectation.expected_claims
        }
        observed_expectation_ids = {
            claim.expectation_id
            for claim in observation.claims
            if claim.expectation_id is not None
        }
        _append_issue(
            issues,
            bool(observed_expectation_ids - expected_claim_ids),
            "UNKNOWN_EXPECTATION_ID",
        )

        expected_kind = case.expectation.answer_kind
        kind_correct: bool | None = None
        refusal_correct: bool | None = None
        if expected_kind != "either":
            answer_kind_eligible += 1
            kind_correct = completed and observation.answer_kind == expected_kind
            answer_kind_correct_count += kind_correct
            _append_issue(issues, not kind_correct, "ANSWER_KIND_MISMATCH")
        if expected_kind == "refusal":
            refusal_eligible += 1
            refusal_correct = completed and observation.answer_kind == "refusal"
            refusal_correct_count += refusal_correct
            _append_issue(issues, not refusal_correct, "REFUSAL_EXPECTATION_FAILED")

        trace_pass = _trace_compliant(case, observation)
        if trace_pass is not None:
            trace_eligible += 1
            trace_compliant_count += trace_pass
            _append_issue(issues, not trace_pass, "TRACE_EXPECTATION_FAILED")

        if observation.latency_ms is not None:
            latencies.append(observation.latency_ms)
        if observation.cost_usd is not None:
            costs.append(observation.cost_usd)
        total_input_tokens += observation.input_tokens or 0
        total_output_tokens += observation.output_tokens or 0

        budget = case.expectation.budget
        latency_pass = _budget_pass(
            observation.latency_ms, budget.max_latency_ms if budget else None
        )
        cost_pass = _budget_pass(
            observation.cost_usd, budget.max_cost_usd if budget else None
        )
        input_pass = _budget_pass(
            observation.input_tokens, budget.max_input_tokens if budget else None
        )
        output_pass = _budget_pass(
            observation.output_tokens, budget.max_output_tokens if budget else None
        )
        for key, passed in (
            ("latency", latency_pass),
            ("cost", cost_pass),
            ("input", input_pass),
            ("output", output_pass),
        ):
            if passed is not None:
                budget_counts[key].append(passed)
                _append_issue(issues, not passed, f"{key.upper()}_BUDGET_EXCEEDED")

        per_case.append(
            CaseScore(
                case_id=case.case_id,
                issues=tuple(issues),
                recall_at_k=recalls,
                mrr=mrr,
                citation_validity=citation_validity,
                citation_coverage=citation_coverage,
                expected_claim_coverage=expected_claim_coverage,
                grounding_score=grounding_score,
                answer_kind_correct=kind_correct,
                refusal_correct=refusal_correct,
                trace_compliant=trace_pass,
                latency_budget_passed=latency_pass,
                cost_budget_passed=cost_pass,
                input_token_budget_passed=input_pass,
                output_token_budget_passed=output_pass,
            )
        )

    def budget_rate(key: str) -> float | None:
        values = budget_counts[key]
        return _score(sum(values), len(values))

    citation_validity = _mean(citation_validity_scores)
    citation_coverage = _mean(citation_coverage_scores)
    grounding_score = _score(
        grounding_totals.grounded_expected_claims,
        grounding_totals.addressed_expected_claims,
    )
    if (
        grounding_totals.expected_claims
        and not grounding_totals.addressed_expected_claims
    ):
        grounding_score = 0.0

    return EvaluationResult(
        schema_version=RESULT_SCHEMA_VERSION,
        suite_id=suite.suite_id,
        run_id=observations.run_id,
        system_id=observations.system_id,
        case_ids=case_ids,
        ks=ks,
        retrieval=RetrievalMetrics(
            eligible_cases=len(retrieval_mrrs),
            recall_at_k={str(k): _mean(retrieval_recalls[k]) or 0.0 for k in ks},
            mrr=_mean(retrieval_mrrs),
        ),
        grounding=GroundingMetrics(
            eligible_answer_cases=eligible_answer_cases,
            citation_count=grounding_totals.citations,
            valid_citation_count=grounding_totals.valid_citations,
            citation_validity=citation_validity,
            observed_claim_count=grounding_totals.claims,
            covered_claim_count=grounding_totals.covered_claims,
            citation_coverage=citation_coverage,
            expected_claim_count=grounding_totals.expected_claims,
            addressed_expected_claim_count=grounding_totals.addressed_expected_claims,
            expected_claim_coverage=_score(
                grounding_totals.addressed_expected_claims,
                grounding_totals.expected_claims,
            ),
            grounded_expected_claim_count=grounding_totals.grounded_expected_claims,
            grounding_score=grounding_score,
        ),
        behavior=BehaviorMetrics(
            total_cases=len(suite.cases),
            completed_cases=completed_cases,
            completion_rate=_score(completed_cases, len(suite.cases)) or 0.0,
            answer_kind_eligible_cases=answer_kind_eligible,
            answer_kind_correct=answer_kind_correct_count,
            answer_kind_accuracy=_score(
                answer_kind_correct_count, answer_kind_eligible
            ),
            refusal_eligible_cases=refusal_eligible,
            refusal_correct=refusal_correct_count,
            refusal_accuracy=_score(refusal_correct_count, refusal_eligible),
            trace_eligible_cases=trace_eligible,
            trace_compliant=trace_compliant_count,
            trace_compliance=_score(trace_compliant_count, trace_eligible),
        ),
        operations=OperationsMetrics(
            latency_ms=_distribution(latencies),
            cost_observation_count=len(costs),
            total_cost_usd=round(sum(costs), 12) if costs else None,
            mean_cost_usd=_mean(costs),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            latency_budget_eligible_cases=len(budget_counts["latency"]),
            latency_budget_pass_rate=budget_rate("latency"),
            cost_budget_eligible_cases=len(budget_counts["cost"]),
            cost_budget_pass_rate=budget_rate("cost"),
            input_token_budget_eligible_cases=len(budget_counts["input"]),
            input_token_budget_pass_rate=budget_rate("input"),
            output_token_budget_eligible_cases=len(budget_counts["output"]),
            output_token_budget_pass_rate=budget_rate("output"),
        ),
        cases=tuple(per_case),
        metadata={
            "case_schema_version": suite.schema_version,
            "observation_schema_version": observations.schema_version,
            "suite_metadata": suite.metadata,
            "run_metadata": observations.metadata,
        },
    )
