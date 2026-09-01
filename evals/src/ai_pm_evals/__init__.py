"""Versioned, deterministic evaluation contracts for AI-PM."""

from .compare import compare_results, default_gate_policy
from .metrics import evaluate
from .models import (
    CASE_SUITE_SCHEMA_VERSION,
    COMPARISON_SCHEMA_VERSION,
    GATE_POLICY_SCHEMA_VERSION,
    OBSERVATION_SET_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    CaseSuite,
    ComparisonReport,
    EvaluationResult,
    GatePolicy,
    ObservationSet,
)

__all__ = [
    "CASE_SUITE_SCHEMA_VERSION",
    "COMPARISON_SCHEMA_VERSION",
    "GATE_POLICY_SCHEMA_VERSION",
    "OBSERVATION_SET_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "CaseSuite",
    "ComparisonReport",
    "EvaluationResult",
    "GatePolicy",
    "ObservationSet",
    "compare_results",
    "default_gate_policy",
    "evaluate",
]
