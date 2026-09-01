from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

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

ModelT = TypeVar("ModelT", bound=BaseModel)

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    CASE_SUITE_SCHEMA_VERSION: CaseSuite,
    OBSERVATION_SET_SCHEMA_VERSION: ObservationSet,
    RESULT_SCHEMA_VERSION: EvaluationResult,
    GATE_POLICY_SCHEMA_VERSION: GatePolicy,
    COMPARISON_SCHEMA_VERSION: ComparisonReport,
}

SCHEMA_FILENAMES: dict[str, str] = {
    CASE_SUITE_SCHEMA_VERSION: "case-suite.v1.schema.json",
    OBSERVATION_SET_SCHEMA_VERSION: "observation-set.v1.schema.json",
    RESULT_SCHEMA_VERSION: "result.v1.schema.json",
    GATE_POLICY_SCHEMA_VERSION: "gate-policy.v1.schema.json",
    COMPARISON_SCHEMA_VERSION: "comparison.v1.schema.json",
}


class EvalInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvalInputError("FILE_READ_ERROR", f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvalInputError(
            "INVALID_JSON", f"invalid JSON in {path} at line {exc.lineno}: {exc.msg}"
        ) from exc


def load_model(path: Path, model: type[ModelT]) -> ModelT:
    return model.model_validate(read_json(path))


def load_versioned(path: Path) -> BaseModel:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise EvalInputError(
            "INVALID_DOCUMENT", "top-level JSON value must be an object"
        )
    version = payload.get("schema_version")
    if not isinstance(version, str):
        raise EvalInputError("MISSING_SCHEMA_VERSION", "schema_version is required")
    model = SCHEMA_MODELS.get(version)
    if model is None:
        raise EvalInputError(
            "UNKNOWN_SCHEMA_VERSION", f"unsupported schema_version: {version}"
        )
    return model.model_validate(payload)


def canonical_json(model: BaseModel) -> str:
    payload = model.model_dump(mode="json", exclude_none=False)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(model), encoding="utf-8")


def write_schemas(output_dir: Path) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for version, filename in SCHEMA_FILENAMES.items():
        model = SCHEMA_MODELS[version]
        target = output_dir / filename
        schema = model.model_json_schema(mode="validation")
        target.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(target)
    return tuple(written)
