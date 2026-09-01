from __future__ import annotations

import json

from ai_pm_evals.cli import main
from support import EVALS_ROOT


def test_validate_and_run_emit_json(tmp_path, capsys):
    case_path = EVALS_ROOT / "cases" / "smoke.v1.json"
    observations_path = EVALS_ROOT / "observations" / "smoke-baseline.v1.json"
    assert main(["validate", str(case_path)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    output = tmp_path / "result.json"
    assert (
        main(
            [
                "run",
                "--cases",
                str(case_path),
                "--observations",
                str(observations_path),
                "--output",
                str(output),
                "--ks",
                "1,3,5",
            ]
        )
        == 0
    )
    stdout_result = json.loads(capsys.readouterr().out)
    assert stdout_result == json.loads(output.read_text(encoding="utf-8"))
    assert stdout_result["schema_version"] == "ai-pm.eval.result.v1"


def test_compare_returns_three_and_writes_failure_report(tmp_path, capsys):
    cases = EVALS_ROOT / "cases" / "smoke.v1.json"
    baseline_observations = EVALS_ROOT / "observations" / "smoke-baseline.v1.json"
    candidate_observations = EVALS_ROOT / "observations" / "smoke-regressed.v1.json"
    baseline_result = tmp_path / "baseline.json"
    candidate_result = tmp_path / "candidate.json"
    comparison_result = tmp_path / "comparison.json"
    for observations, output in (
        (baseline_observations, baseline_result),
        (candidate_observations, candidate_result),
    ):
        assert (
            main(
                [
                    "run",
                    "--cases",
                    str(cases),
                    "--observations",
                    str(observations),
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        capsys.readouterr()

    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline_result),
            "--candidate",
            str(candidate_result),
            "--policy",
            str(EVALS_ROOT / "policies" / "non-regression.v1.json"),
            "--output",
            str(comparison_result),
        ]
    )
    assert exit_code == 3
    assert json.loads(capsys.readouterr().out)["passed"] is False
    assert json.loads(comparison_result.read_text(encoding="utf-8"))["passed"] is False


def test_invalid_json_returns_machine_readable_error(tmp_path, capsys):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert main(["validate", str(invalid)]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "INVALID_JSON"
