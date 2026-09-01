from __future__ import annotations

import pytest
from ai_pm_evals.io import EvalInputError
from ai_pm_evals.metrics import evaluate
from support import (
    load_baseline_observations,
    load_regressed_observations,
    load_smoke_suite,
)


def test_baseline_metrics_are_deterministic():
    smoke_suite = load_smoke_suite()
    baseline_observations = load_baseline_observations()
    first = evaluate(smoke_suite, baseline_observations, (1, 3, 5))
    second = evaluate(smoke_suite, baseline_observations, (1, 3, 5))
    assert first == second
    assert first.retrieval.eligible_cases == 2
    assert first.retrieval.recall_at_k == {
        "1": pytest.approx(0.416666666666),
        "3": pytest.approx(0.833333333334),
        "5": 1.0,
    }
    assert first.retrieval.mrr == 1.0
    assert first.grounding.citation_validity == 1.0
    assert first.grounding.citation_coverage == 1.0
    assert first.grounding.expected_claim_coverage == 1.0
    assert first.grounding.grounding_score == 1.0
    assert first.behavior.refusal_accuracy == 1.0
    assert first.behavior.trace_compliance == 1.0
    assert first.behavior.completion_rate == 1.0
    assert first.operations.latency_ms.p95 == 1350
    assert first.operations.total_cost_usd == 0.014
    assert not any(score.issues for score in first.cases)


def test_regressed_run_exposes_independent_failures():
    smoke_suite = load_smoke_suite()
    regressed_observations = load_regressed_observations()
    result = evaluate(smoke_suite, regressed_observations, (1, 3, 5))
    assert result.retrieval.recall_at_k["5"] < 1
    assert result.grounding.citation_validity < 1
    assert result.grounding.citation_coverage < 1
    assert result.grounding.expected_claim_coverage < 1
    assert result.behavior.refusal_accuracy == 0
    assert result.behavior.completion_rate < 1
    assert result.behavior.trace_compliance < 1
    issues = {issue for case in result.cases for issue in case.issues}
    assert "INVALID_OR_MISSING_CITATION" in issues
    assert "REFUSAL_EXPECTATION_FAILED" in issues
    assert "TRACE_EXPECTATION_FAILED" in issues
    assert "OUTCOME_TRUNCATED" in issues


def test_case_set_and_ks_must_match():
    smoke_suite = load_smoke_suite()
    baseline_observations = load_baseline_observations()
    shortened = baseline_observations.model_copy(
        update={"cases": baseline_observations.cases[:-1]}
    )
    with pytest.raises(EvalInputError, match="case set mismatch"):
        evaluate(smoke_suite, shortened)
    with pytest.raises(EvalInputError, match="unique positive integers"):
        evaluate(smoke_suite, baseline_observations, (1, 1))
