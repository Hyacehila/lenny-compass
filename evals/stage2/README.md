# Stage 2 evaluation pack

This directory contains deterministic, synthetic inputs for the question planner
and claim-to-evidence verifier. It contains no paid corpus text.

- `question-planning.v1.json` has 60 knowledge, comparison, scenario-advice, and
  follow-up cases. It records whether one clarification is warranted.
- `claim-support.v1.json` has 200 atomic claim/evidence pairs balanced across
  supported, insufficient-context, and contradicted cases.
- `human-review.v1.json` is the required review sheet: 20 claim/evidence pairs and
  eight end-to-end product scenarios. It intentionally starts incomplete.

Regenerate the three files deterministically from the repository root:

```powershell
uv run python evals/tools/generate_stage2_cases.py
```

The UI must continue to say that human calibration is pending until a person fills
every verdict and end-to-end rating, adds their name and review time, and explicitly
approves the pack. Automatic tests or model-generated labels do not count as human
calibration.
