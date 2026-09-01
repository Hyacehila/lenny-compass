from __future__ import annotations

import json

import pytest
from ai_pm_evals.io import SCHEMA_FILENAMES, SCHEMA_MODELS, load_versioned
from ai_pm_evals.models import CaseSuite
from pydantic import ValidationError
from support import EVALS_ROOT, load_smoke_suite


def test_all_checked_in_documents_validate():
    paths = (
        EVALS_ROOT / "cases" / "smoke.v1.json",
        EVALS_ROOT / "observations" / "smoke-baseline.v1.json",
        EVALS_ROOT / "observations" / "smoke-regressed.v1.json",
        EVALS_ROOT / "policies" / "non-regression.v1.json",
    )
    for path in paths:
        assert load_versioned(path).schema_version


def test_case_contract_rejects_unknown_fields_and_duplicate_ids():
    smoke_suite = load_smoke_suite()
    payload = smoke_suite.model_dump(mode="json")
    payload["unknown"] = True
    with pytest.raises(ValidationError):
        CaseSuite.model_validate(payload)

    payload.pop("unknown")
    payload["cases"].append(payload["cases"][0])
    with pytest.raises(ValidationError, match="duplicate case ids"):
        CaseSuite.model_validate(payload)


def test_checked_in_json_schemas_match_models():
    schema_root = EVALS_ROOT / "schemas"
    for version, filename in SCHEMA_FILENAMES.items():
        actual = json.loads((schema_root / filename).read_text(encoding="utf-8"))
        assert actual == SCHEMA_MODELS[version].model_json_schema(mode="validation")
