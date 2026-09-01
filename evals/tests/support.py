from __future__ import annotations

from pathlib import Path

from ai_pm_evals.io import load_model
from ai_pm_evals.models import CaseSuite, ObservationSet

EVALS_ROOT = Path(__file__).resolve().parents[1]


def load_smoke_suite() -> CaseSuite:
    return load_model(EVALS_ROOT / "cases" / "smoke.v1.json", CaseSuite)


def load_baseline_observations() -> ObservationSet:
    return load_model(
        EVALS_ROOT / "observations" / "smoke-baseline.v1.json", ObservationSet
    )


def load_regressed_observations() -> ObservationSet:
    return load_model(
        EVALS_ROOT / "observations" / "smoke-regressed.v1.json", ObservationSet
    )
