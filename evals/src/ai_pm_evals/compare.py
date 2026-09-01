from __future__ import annotations

from typing import Any

from .models import (
    COMPARISON_SCHEMA_VERSION,
    GATE_POLICY_SCHEMA_VERSION,
    ComparisonCheck,
    ComparisonReport,
    EvaluationResult,
    GateCheckSpec,
    GatePolicy,
)


def default_gate_policy() -> GatePolicy:
    quality_metrics = (
        "retrieval.recall_at_k.5",
        "retrieval.mrr",
        "grounding.citation_validity",
        "grounding.citation_coverage",
        "grounding.expected_claim_coverage",
        "grounding.grounding_score",
        "behavior.completion_rate",
        "behavior.answer_kind_accuracy",
        "behavior.refusal_accuracy",
        "behavior.trace_compliance",
    )
    checks = [
        GateCheckSpec(
            check_id=metric.replace(".", "-"),
            metric=metric,
            rule="non_regression",
            direction="higher",
        )
        for metric in quality_metrics
    ]
    checks.extend(
        (
            GateCheckSpec(
                check_id="latency-p95",
                metric="operations.latency_ms.p95",
                rule="non_regression",
                direction="lower",
                relative_tolerance=0.20,
            ),
            GateCheckSpec(
                check_id="mean-cost-usd",
                metric="operations.mean_cost_usd",
                rule="non_regression",
                direction="lower",
                relative_tolerance=0.20,
            ),
        )
    )
    return GatePolicy(
        schema_version=GATE_POLICY_SCHEMA_VERSION,
        policy_id="default-non-regression-v1",
        checks=tuple(checks),
    )


def _metric(result: EvaluationResult, path: str) -> float | None:
    value: Any = result.model_dump(mode="json")
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _availability_check(
    spec: GateCheckSpec, baseline: float | None, candidate: float | None
) -> ComparisonCheck | None:
    if candidate is not None and (
        spec.rule != "non_regression" or baseline is not None
    ):
        return None
    unavailable = []
    if spec.rule == "non_regression" and baseline is None:
        unavailable.append("baseline")
    if candidate is None:
        unavailable.append("candidate")
    status = "skipped" if spec.when_unavailable == "skip" else "failed"
    if baseline is not None and candidate is None:
        status = "failed"
    return ComparisonCheck(
        check_id=spec.check_id,
        metric=spec.metric,
        rule=spec.rule,
        status=status,
        baseline_value=baseline,
        candidate_value=candidate,
        message=f"metric unavailable in {', '.join(unavailable)} result",
    )


def _evaluate_check(
    spec: GateCheckSpec, baseline: float | None, candidate: float | None
) -> ComparisonCheck:
    unavailable = _availability_check(spec, baseline, candidate)
    if unavailable is not None:
        return unavailable
    assert candidate is not None
    if spec.rule == "minimum":
        assert spec.value is not None
        passed = candidate >= spec.value
        message = f"candidate {candidate:g} must be >= {spec.value:g}"
        allowed_delta = None
    elif spec.rule == "maximum":
        assert spec.value is not None
        passed = candidate <= spec.value
        message = f"candidate {candidate:g} must be <= {spec.value:g}"
        allowed_delta = None
    else:
        assert baseline is not None and spec.direction is not None
        allowed_delta = max(
            spec.absolute_tolerance, abs(baseline) * spec.relative_tolerance
        )
        if spec.direction == "higher":
            passed = candidate >= baseline - allowed_delta
            message = (
                f"candidate {candidate:g} must be >= baseline {baseline:g} "
                f"minus tolerance {allowed_delta:g}"
            )
        else:
            passed = candidate <= baseline + allowed_delta
            message = (
                f"candidate {candidate:g} must be <= baseline {baseline:g} "
                f"plus tolerance {allowed_delta:g}"
            )
    return ComparisonCheck(
        check_id=spec.check_id,
        metric=spec.metric,
        rule=spec.rule,
        status="passed" if passed else "failed",
        baseline_value=baseline,
        candidate_value=candidate,
        allowed_delta=allowed_delta,
        message=message,
    )


def compare_results(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    policy: GatePolicy | None = None,
) -> ComparisonReport:
    policy = policy or default_gate_policy()
    checks: list[ComparisonCheck] = []
    if policy.require_same_suite:
        passed = baseline.suite_id == candidate.suite_id
        checks.append(
            ComparisonCheck(
                check_id="same-suite",
                metric="suite_id",
                rule="equality",
                status="passed" if passed else "failed",
                message=(
                    f"baseline suite {baseline.suite_id!r} and candidate suite "
                    f"{candidate.suite_id!r} must match"
                ),
            )
        )
    if policy.require_same_case_set:
        passed = set(baseline.case_ids) == set(candidate.case_ids)
        checks.append(
            ComparisonCheck(
                check_id="same-case-set",
                metric="case_ids",
                rule="set_equality",
                status="passed" if passed else "failed",
                message="baseline and candidate case sets must match",
            )
        )
    for spec in policy.checks:
        checks.append(
            _evaluate_check(
                spec, _metric(baseline, spec.metric), _metric(candidate, spec.metric)
            )
        )
    return ComparisonReport(
        schema_version=COMPARISON_SCHEMA_VERSION,
        policy_id=policy.policy_id,
        baseline_suite_id=baseline.suite_id,
        candidate_suite_id=candidate.suite_id,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        passed=all(check.status != "failed" for check in checks),
        checks=tuple(checks),
    )
