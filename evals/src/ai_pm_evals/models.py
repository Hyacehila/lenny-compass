from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

CASE_SUITE_SCHEMA_VERSION = "ai-pm.eval.case-suite.v1"
OBSERVATION_SET_SCHEMA_VERSION = "ai-pm.eval.observation-set.v1"
RESULT_SCHEMA_VERSION = "ai-pm.eval.result.v1"
GATE_POLICY_SCHEMA_VERSION = "ai-pm.eval.gate-policy.v1"
COMPARISON_SCHEMA_VERSION = "ai-pm.eval.comparison.v1"

Identifier = Annotated[str, Field(min_length=1, max_length=160)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
UnitScore = Annotated[float, Field(ge=0, le=1)]


class EvalModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


def _duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


class ExpectedClaim(EvalModel):
    claim_id: Identifier
    acceptable_evidence_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> ExpectedClaim:
        duplicates = _duplicates(self.acceptable_evidence_ids)
        if duplicates:
            raise ValueError(f"duplicate acceptable evidence ids: {duplicates}")
        return self


class TraceExpectation(EvalModel):
    required_tools: tuple[Identifier, ...] = ()
    forbidden_tools: tuple[Identifier, ...] = ()
    max_steps: NonNegativeInt | None = None
    require_completed: bool = True

    @model_validator(mode="after")
    def tool_sets_are_valid(self) -> TraceExpectation:
        duplicate_required = _duplicates(self.required_tools)
        duplicate_forbidden = _duplicates(self.forbidden_tools)
        if duplicate_required or duplicate_forbidden:
            raise ValueError("trace tool lists must not contain duplicates")
        overlap = sorted(set(self.required_tools) & set(self.forbidden_tools))
        if overlap:
            raise ValueError(f"tools cannot be both required and forbidden: {overlap}")
        return self


class BudgetExpectation(EvalModel):
    max_latency_ms: NonNegativeFloat | None = None
    max_cost_usd: NonNegativeFloat | None = None
    max_input_tokens: NonNegativeInt | None = None
    max_output_tokens: NonNegativeInt | None = None


class CaseExpectation(EvalModel):
    relevant_evidence_ids: tuple[Identifier, ...] = ()
    answer_kind: Literal["answer", "refusal", "either"] = "answer"
    expected_claims: tuple[ExpectedClaim, ...] = ()
    trace: TraceExpectation | None = None
    budget: BudgetExpectation | None = None

    @model_validator(mode="after")
    def expectations_are_consistent(self) -> CaseExpectation:
        duplicate_evidence = _duplicates(self.relevant_evidence_ids)
        if duplicate_evidence:
            raise ValueError(f"duplicate relevant evidence ids: {duplicate_evidence}")
        claim_ids = tuple(claim.claim_id for claim in self.expected_claims)
        duplicate_claims = _duplicates(claim_ids)
        if duplicate_claims:
            raise ValueError(f"duplicate expected claim ids: {duplicate_claims}")
        if self.answer_kind == "refusal" and self.expected_claims:
            raise ValueError("refusal cases cannot define expected claims")
        return self


class EvaluationCase(EvalModel):
    case_id: Identifier
    query: Annotated[str, Field(min_length=1, max_length=4000)]
    tags: tuple[Identifier, ...] = ()
    expectation: CaseExpectation
    notes: Annotated[str, Field(max_length=1000)] | None = None

    @model_validator(mode="after")
    def tags_are_unique(self) -> EvaluationCase:
        duplicates = _duplicates(self.tags)
        if duplicates:
            raise ValueError(f"duplicate tags: {duplicates}")
        return self


class CaseSuite(EvalModel):
    schema_version: Literal[CASE_SUITE_SCHEMA_VERSION]
    suite_id: Identifier
    description: Annotated[str, Field(min_length=1, max_length=1000)]
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> CaseSuite:
        duplicates = _duplicates(tuple(case.case_id for case in self.cases))
        if duplicates:
            raise ValueError(f"duplicate case ids: {duplicates}")
        return self


class ClaimObservation(EvalModel):
    claim_id: Identifier
    expectation_id: Identifier | None = None
    citation_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def citations_are_unique(self) -> ClaimObservation:
        duplicates = _duplicates(self.citation_ids)
        if duplicates:
            raise ValueError(f"duplicate citation ids: {duplicates}")
        return self


class ToolEvent(EvalModel):
    tool: Identifier
    status: Literal["ok", "error", "cancelled"] = "ok"
    latency_ms: NonNegativeFloat | None = None


class TraceObservation(EvalModel):
    events: tuple[ToolEvent, ...] = ()
    completed: bool
    termination_reason: Identifier | None = None


class CaseObservation(EvalModel):
    case_id: Identifier
    outcome: Literal["completed", "failed", "cancelled", "truncated"] = "completed"
    retrieved_evidence_ids: tuple[Identifier, ...] = ()
    grounded_evidence_ids: tuple[Identifier, ...] = ()
    answer_kind: Literal["answer", "refusal"] | None = None
    claims: tuple[ClaimObservation, ...] = ()
    trace: TraceObservation | None = None
    latency_ms: NonNegativeFloat | None = None
    cost_usd: NonNegativeFloat | None = None
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None

    @model_validator(mode="after")
    def observation_is_consistent(self) -> CaseObservation:
        for field_name, values in (
            ("retrieved_evidence_ids", self.retrieved_evidence_ids),
            ("grounded_evidence_ids", self.grounded_evidence_ids),
            ("claim ids", tuple(claim.claim_id for claim in self.claims)),
        ):
            duplicates = _duplicates(values)
            if duplicates:
                raise ValueError(f"duplicate {field_name}: {duplicates}")
        if self.outcome == "completed" and self.answer_kind is None:
            raise ValueError("completed observations require answer_kind")
        if self.answer_kind == "refusal" and self.claims:
            raise ValueError("refusal observations cannot contain claims")
        expectation_ids = tuple(
            claim.expectation_id
            for claim in self.claims
            if claim.expectation_id is not None
        )
        duplicates = _duplicates(expectation_ids)
        if duplicates:
            raise ValueError(f"duplicate claim expectation ids: {duplicates}")
        return self


class ObservationSet(EvalModel):
    schema_version: Literal[OBSERVATION_SET_SCHEMA_VERSION]
    run_id: Identifier
    suite_id: Identifier
    system_id: Identifier
    cases: tuple[CaseObservation, ...] = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> ObservationSet:
        duplicates = _duplicates(tuple(case.case_id for case in self.cases))
        if duplicates:
            raise ValueError(f"duplicate observation case ids: {duplicates}")
        return self


class RetrievalMetrics(EvalModel):
    eligible_cases: NonNegativeInt
    recall_at_k: dict[str, UnitScore]
    mrr: UnitScore | None

    @model_validator(mode="after")
    def eligibility_is_consistent(self) -> RetrievalMetrics:
        if self.eligible_cases == 0 and self.mrr is not None:
            raise ValueError("mrr must be null when no retrieval cases are eligible")
        if self.eligible_cases > 0 and self.mrr is None:
            raise ValueError("mrr is required when retrieval cases are eligible")
        return self


class GroundingMetrics(EvalModel):
    eligible_answer_cases: NonNegativeInt
    citation_count: NonNegativeInt
    valid_citation_count: NonNegativeInt
    citation_validity: UnitScore | None
    observed_claim_count: NonNegativeInt
    covered_claim_count: NonNegativeInt
    citation_coverage: UnitScore | None
    expected_claim_count: NonNegativeInt
    addressed_expected_claim_count: NonNegativeInt
    expected_claim_coverage: UnitScore | None
    grounded_expected_claim_count: NonNegativeInt
    grounding_score: UnitScore | None

    @model_validator(mode="after")
    def counts_are_consistent(self) -> GroundingMetrics:
        if self.valid_citation_count > self.citation_count:
            raise ValueError("valid citation count cannot exceed citation count")
        if self.covered_claim_count > self.observed_claim_count:
            raise ValueError("covered claim count cannot exceed observed claim count")
        if self.addressed_expected_claim_count > self.expected_claim_count:
            raise ValueError("addressed expected claims cannot exceed expected claims")
        if self.grounded_expected_claim_count > self.addressed_expected_claim_count:
            raise ValueError("grounded expected claims cannot exceed addressed claims")
        return self


class BehaviorMetrics(EvalModel):
    total_cases: NonNegativeInt
    completed_cases: NonNegativeInt
    completion_rate: UnitScore
    answer_kind_eligible_cases: NonNegativeInt
    answer_kind_correct: NonNegativeInt
    answer_kind_accuracy: UnitScore | None
    refusal_eligible_cases: NonNegativeInt
    refusal_correct: NonNegativeInt
    refusal_accuracy: UnitScore | None
    trace_eligible_cases: NonNegativeInt
    trace_compliant: NonNegativeInt
    trace_compliance: UnitScore | None

    @model_validator(mode="after")
    def counts_are_consistent(self) -> BehaviorMetrics:
        pairs = (
            (self.completed_cases, self.total_cases, "completed"),
            (self.answer_kind_correct, self.answer_kind_eligible_cases, "answer kind"),
            (self.refusal_correct, self.refusal_eligible_cases, "refusal"),
            (self.trace_compliant, self.trace_eligible_cases, "trace"),
        )
        for successful, eligible, label in pairs:
            if successful > eligible:
                raise ValueError(f"{label} successes cannot exceed eligible cases")
        return self


class DistributionMetrics(EvalModel):
    count: NonNegativeInt
    mean: NonNegativeFloat | None
    p50: NonNegativeFloat | None
    p95: NonNegativeFloat | None
    max: NonNegativeFloat | None

    @model_validator(mode="after")
    def values_match_count(self) -> DistributionMetrics:
        values = (self.mean, self.p50, self.p95, self.max)
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("empty distributions cannot contain values")
        if self.count > 0 and any(value is None for value in values):
            raise ValueError("non-empty distributions require all summary values")
        return self


class OperationsMetrics(EvalModel):
    latency_ms: DistributionMetrics
    cost_observation_count: NonNegativeInt
    total_cost_usd: NonNegativeFloat | None
    mean_cost_usd: NonNegativeFloat | None
    total_input_tokens: NonNegativeInt
    total_output_tokens: NonNegativeInt
    latency_budget_eligible_cases: NonNegativeInt
    latency_budget_pass_rate: UnitScore | None
    cost_budget_eligible_cases: NonNegativeInt
    cost_budget_pass_rate: UnitScore | None
    input_token_budget_eligible_cases: NonNegativeInt
    input_token_budget_pass_rate: UnitScore | None
    output_token_budget_eligible_cases: NonNegativeInt
    output_token_budget_pass_rate: UnitScore | None


class CaseScore(EvalModel):
    case_id: Identifier
    issues: tuple[Identifier, ...]
    recall_at_k: dict[str, UnitScore] | None
    mrr: UnitScore | None
    citation_validity: UnitScore | None
    citation_coverage: UnitScore | None
    expected_claim_coverage: UnitScore | None
    grounding_score: UnitScore | None
    answer_kind_correct: bool | None
    refusal_correct: bool | None
    trace_compliant: bool | None
    latency_budget_passed: bool | None
    cost_budget_passed: bool | None
    input_token_budget_passed: bool | None
    output_token_budget_passed: bool | None


class EvaluationResult(EvalModel):
    schema_version: Literal[RESULT_SCHEMA_VERSION]
    suite_id: Identifier
    run_id: Identifier
    system_id: Identifier
    case_ids: tuple[Identifier, ...]
    ks: tuple[Annotated[int, Field(gt=0)], ...] = Field(min_length=1)
    retrieval: RetrievalMetrics
    grounding: GroundingMetrics
    behavior: BehaviorMetrics
    operations: OperationsMetrics
    cases: tuple[CaseScore, ...]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def result_is_internally_consistent(self) -> EvaluationResult:
        if _duplicates(self.case_ids):
            raise ValueError("result case_ids must be unique")
        if len(set(self.ks)) != len(self.ks) or tuple(sorted(self.ks)) != self.ks:
            raise ValueError("result ks must be unique and sorted")
        scored_ids = tuple(score.case_id for score in self.cases)
        if scored_ids != self.case_ids:
            raise ValueError("case score ids and order must exactly match case_ids")
        expected_recall_keys = {str(k) for k in self.ks}
        if set(self.retrieval.recall_at_k) != expected_recall_keys:
            raise ValueError("aggregate recall keys must exactly match ks")
        for score in self.cases:
            if (
                score.recall_at_k is not None
                and set(score.recall_at_k) != expected_recall_keys
            ):
                raise ValueError(
                    f"recall keys for {score.case_id} must exactly match ks"
                )
        if self.retrieval.eligible_cases > len(self.case_ids):
            raise ValueError("retrieval eligible cases cannot exceed total cases")
        if self.behavior.total_cases != len(self.case_ids):
            raise ValueError("behavior total_cases must match case_ids")
        return self


class GateCheckSpec(EvalModel):
    check_id: Identifier
    metric: Identifier
    rule: Literal["non_regression", "minimum", "maximum"]
    direction: Literal["higher", "lower"] | None = None
    value: NonNegativeFloat | None = None
    absolute_tolerance: NonNegativeFloat = 0
    relative_tolerance: NonNegativeFloat = 0
    when_unavailable: Literal["fail", "skip"] = "skip"

    @model_validator(mode="after")
    def rule_has_required_fields(self) -> GateCheckSpec:
        if self.rule == "non_regression":
            if self.direction is None:
                raise ValueError("non_regression checks require direction")
            if self.value is not None:
                raise ValueError("non_regression checks cannot define value")
        else:
            if self.value is None:
                raise ValueError(f"{self.rule} checks require value")
            if self.direction is not None:
                raise ValueError(f"{self.rule} checks cannot define direction")
            if self.absolute_tolerance or self.relative_tolerance:
                raise ValueError(f"{self.rule} checks cannot define tolerances")
        return self


class GatePolicy(EvalModel):
    schema_version: Literal[GATE_POLICY_SCHEMA_VERSION]
    policy_id: Identifier
    require_same_suite: bool = True
    require_same_case_set: bool = True
    checks: tuple[GateCheckSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_ids_are_unique(self) -> GatePolicy:
        check_ids = tuple(check.check_id for check in self.checks)
        duplicates = _duplicates(check_ids)
        if duplicates:
            raise ValueError(f"duplicate gate check ids: {duplicates}")
        reserved = set()
        if self.require_same_suite:
            reserved.add("same-suite")
        if self.require_same_case_set:
            reserved.add("same-case-set")
        overlap = sorted(reserved & set(check_ids))
        if overlap:
            raise ValueError(f"gate check ids use reserved structural ids: {overlap}")
        return self


class ComparisonCheck(EvalModel):
    check_id: Identifier
    metric: Identifier
    rule: Identifier
    status: Literal["passed", "failed", "skipped"]
    baseline_value: float | None = None
    candidate_value: float | None = None
    allowed_delta: NonNegativeFloat | None = None
    message: Annotated[str, Field(min_length=1, max_length=1000)]


class ComparisonReport(EvalModel):
    schema_version: Literal[COMPARISON_SCHEMA_VERSION]
    policy_id: Identifier
    baseline_suite_id: Identifier
    candidate_suite_id: Identifier
    baseline_run_id: Identifier
    candidate_run_id: Identifier
    passed: bool
    checks: tuple[ComparisonCheck, ...]

    @model_validator(mode="after")
    def report_is_consistent(self) -> ComparisonReport:
        if _duplicates(tuple(check.check_id for check in self.checks)):
            raise ValueError("comparison check ids must be unique")
        expected_pass = all(check.status != "failed" for check in self.checks)
        if self.passed != expected_pass:
            raise ValueError("comparison passed flag does not match check statuses")
        return self
