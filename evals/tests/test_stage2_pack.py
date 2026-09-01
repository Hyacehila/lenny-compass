from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(name: str):
    return json.loads((ROOT / "evals" / "stage2" / name).read_text(encoding="utf-8"))


def test_question_planning_pack_has_required_coverage() -> None:
    payload = _read("question-planning.v1.json")
    cases = payload["cases"]
    assert payload["schema_version"] == "stage2-question-planning.v1"
    assert len(cases) == 60
    assert len({item["case_id"] for item in cases}) == 60
    counts = Counter(item["expected_intent"] for item in cases)
    assert counts == {
        "knowledge": 15,
        "comparison": 15,
        "scenario_advice": 20,
        "follow_up": 10,
    }
    assert sum(item["expected_clarification"] for item in cases) == 10


def test_claim_support_pack_is_balanced_and_atomic() -> None:
    payload = _read("claim-support.v1.json")
    cases = payload["cases"]
    assert payload["schema_version"] == "stage2-claim-support.v1"
    assert len(cases) == 200
    assert len({item["case_id"] for item in cases}) == 200
    assert Counter(item["expected_verdict"] for item in cases) == {
        "supported": 75,
        "insufficient_context": 75,
        "contradicted": 50,
    }
    for item in cases:
        assert item["claim"]["evidence_ids"] == [item["evidence"]["evidence_id"]]
        assert "\n" not in item["claim"]["text"]


def test_human_review_pack_stays_honestly_pending_until_completed() -> None:
    payload = _read("human-review.v1.json")
    assert payload["status"] == "awaiting_human_review"
    assert len(payload["claim_pairs"]) == 20
    assert len(payload["e2e_cases"]) == 8
    assert all(item["human_verdict"] is None for item in payload["claim_pairs"])
    assert all(item["rating"] is None for item in payload["e2e_cases"])
    assert payload["reviewer"] == ""
    assert payload["reviewed_at"] is None
