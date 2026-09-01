from __future__ import annotations

from ai_pm_evals.compare import compare_results, default_gate_policy
from ai_pm_evals.metrics import evaluate
from ai_pm_evals.models import GATE_POLICY_SCHEMA_VERSION, GateCheckSpec, GatePolicy
from support import (
    load_baseline_observations,
    load_regressed_observations,
    load_smoke_suite,
)


def test_default_non_regression_gate_passes_identical_result():
    smoke_suite = load_smoke_suite()
    baseline_observations = load_baseline_observations()
    result = evaluate(smoke_suite, baseline_observations)
    report = compare_results(result, result)
    assert report.passed
    assert all(check.status in {"passed", "skipped"} for check in report.checks)


def test_non_regression_gate_fails_regressed_result():
    smoke_suite = load_smoke_suite()
    baseline_observations = load_baseline_observations()
    regressed_observations = load_regressed_observations()
    baseline = evaluate(smoke_suite, baseline_observations)
    candidate = evaluate(smoke_suite, regressed_observations)
    report = compare_results(baseline, candidate, default_gate_policy())
    assert not report.passed
    failed = {check.metric for check in report.checks if check.status == "failed"}
    assert "retrieval.recall_at_k.5" in failed
    assert "grounding.citation_validity" in failed
    assert "behavior.refusal_accuracy" in failed


def test_threshold_policy_does_not_need_a_baseline_metric():
    smoke_suite = load_smoke_suite()
    baseline_observations = load_baseline_observations()
    result = evaluate(smoke_suite, baseline_observations)
    policy = GatePolicy(
        schema_version=GATE_POLICY_SCHEMA_VERSION,
        policy_id="minimum-completion",
        checks=(
            GateCheckSpec(
                check_id="completion",
                metric="behavior.completion_rate",
                rule="minimum",
                value=1,
            ),
        ),
    )
    assert compare_results(result, result, policy).passed
