from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .compare import compare_results, default_gate_policy
from .io import (
    EvalInputError,
    canonical_json,
    load_model,
    load_versioned,
    write_model,
    write_schemas,
)
from .metrics import evaluate
from .models import CaseSuite, EvaluationResult, GatePolicy, ObservationSet


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-pm-eval",
        description="Run deterministic AI-PM offline evaluations and quality gates.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate a versioned JSON document"
    )
    validate.add_argument("path", type=Path)

    run = subparsers.add_parser(
        "run", help="score an observation set against a case suite"
    )
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--observations", type=Path, required=True)
    run.add_argument("--output", type=Path)
    run.add_argument(
        "--ks", default="1,3,5,10", help="comma-separated Recall@K cutoffs"
    )

    compare = subparsers.add_parser(
        "compare", help="compare candidate and baseline results using a quality gate"
    )
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--policy", type=Path)
    compare.add_argument("--output", type=Path)

    schema = subparsers.add_parser("schema", help="write the supported JSON Schemas")
    schema.add_argument("--output-dir", type=Path, required=True)
    return parser


def _parse_ks(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise EvalInputError("INVALID_KS", "--ks must contain integers") from exc


def _emit(model: object, output: Path | None = None) -> None:
    if hasattr(model, "model_dump"):
        rendered = canonical_json(model)  # type: ignore[arg-type]
        if output is not None:
            write_model(output, model)  # type: ignore[arg-type]
    else:
        rendered = (
            json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    sys.stdout.write(rendered)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            model = load_versioned(args.path)
            _emit(
                {
                    "path": str(args.path),
                    "schema_version": model.schema_version,
                    "valid": True,
                }
            )
            return 0
        if args.command == "schema":
            paths = write_schemas(args.output_dir)
            _emit({"schemas": [str(path) for path in paths], "valid": True})
            return 0
        if args.command == "run":
            suite = load_model(args.cases, CaseSuite)
            observations = load_model(args.observations, ObservationSet)
            result = evaluate(suite, observations, _parse_ks(args.ks))
            _emit(result, args.output)
            return 0
        baseline = load_model(args.baseline, EvaluationResult)
        candidate = load_model(args.candidate, EvaluationResult)
        policy = (
            load_model(args.policy, GatePolicy)
            if args.policy
            else default_gate_policy()
        )
        comparison = compare_results(baseline, candidate, policy)
        _emit(comparison, args.output)
        return 0 if comparison.passed else 3
    except (EvalInputError, OSError, ValidationError, ValueError) as exc:
        code = getattr(exc, "code", "VALIDATION_ERROR")
        sys.stderr.write(
            json.dumps(
                {"error": {"code": code, "message": str(exc)}},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        return 2
