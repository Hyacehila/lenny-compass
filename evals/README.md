# AI-PM deterministic evaluations

This package provides a versioned, offline evaluation boundary. It does not call a
model, open the fact store, or persist answer prose. A system adapter records only
ranked evidence IDs, claim-to-citation mappings, answer kind, tool events, latency,
tokens, and cost; this package scores that observation deterministically.

The checked-in smoke suite is entirely synthetic. Its evidence IDs are fictional and
no paid source passage is copied into the repository.

## Contracts

- `case-suite.v1` describes gold retrieval IDs, expected claims and acceptable
  evidence IDs, answer/refusal behavior, trace requirements, and optional budgets.
- `observation-set.v1` records one system run. `grounded_evidence_ids` means evidence
  actually made available to the answer generator; citations outside that set are
  invalid even if they appeared in retrieval.
- `result.v1` contains aggregate and per-case metrics. It contains no generated answer
  text and has no generated timestamp, so identical inputs produce byte-identical JSON.
- `gate-policy.v1` and `comparison.v1` make non-regression decisions machine-readable.

Checked-in JSON Schemas in `schemas/` are generated from the strict Pydantic models.
Unknown fields and duplicate IDs are rejected.

## Metric semantics

- `Recall@K`: fraction of a case's relevant evidence IDs present in the first K
  retrieved IDs, macro-averaged over cases that define relevant evidence.
- `MRR`: reciprocal rank of the first relevant evidence, macro-averaged over the same
  eligible cases. A miss scores zero.
- `citation_validity`: for each observed answer, cited IDs present in
  `grounded_evidence_ids` divided by all cited IDs, macro-averaged over answers. An
  answer with no citations scores zero; raw citation counts are also reported.
- `citation_coverage`: for each observed answer, claims with at least one valid
  citation divided by all claims, macro-averaged over answers. An answer with no
  claims scores zero; raw claim counts are also reported.
- `expected_claim_coverage`: expected claim IDs addressed by an observed claim divided
  by all expected claim IDs.
- `grounding_score`: addressed expected claims citing at least one acceptable,
  grounded evidence ID divided by all addressed expected claims. If expected claims
  exist but none are addressed, it scores zero.
- refusal, completion, answer-kind, trace, latency, token, cost, and per-case budget
  fields are reported independently so a single blended score cannot hide a failure.

Cases without a denominator are excluded and reported as `null`, not silently treated
as perfect. Percentiles use deterministic nearest-rank calculation.

`grounding_score` is an exact, structural claim-to-evidence check rather than an LLM
entailment judge. The producing adapter or a human annotation step must assign each
observed claim's `expectation_id`; unrecognized expectation IDs are surfaced as a
per-case issue. This keeps the core gate deterministic and auditable.
It must not be reported as proof that arbitrary generated prose is semantically
entailed by a cited passage; that requires a separately validated semantic or human
review gate.

## Commands

From the repository root:

```powershell
uv run --package ai-pm-evals ai-pm-eval validate evals/cases/smoke.v1.json

uv run --package ai-pm-evals ai-pm-eval run `
  --cases evals/cases/smoke.v1.json `
  --observations evals/observations/smoke-baseline.v1.json `
  --output evals/results/smoke-baseline.json

uv run --package ai-pm-evals ai-pm-eval run `
  --cases evals/cases/smoke.v1.json `
  --observations evals/observations/smoke-regressed.v1.json `
  --output evals/results/smoke-regressed.json

uv run --package ai-pm-evals ai-pm-eval compare `
  --baseline evals/results/smoke-baseline.json `
  --candidate evals/results/smoke-regressed.json `
  --policy evals/policies/non-regression.v1.json `
  --output evals/results/smoke-comparison.json
```

All commands emit JSON. `validate` and `run` return 0 on success and 2 for invalid
input. `compare` returns 0 when the gate passes, 3 when a check fails, and 2 for
invalid input. The default compare policy is the same quality-oriented policy exposed
by the library; an explicit versioned policy is recommended in CI.

Evaluation results are reproducible build output and remain ignored by Git. Cases,
observations, policies, and schemas are versioned.

To regenerate schemas after a contract change:

```powershell
uv run --package ai-pm-evals ai-pm-eval schema --output-dir evals/schemas
```

Run the package tests with:

```powershell
uv run --package ai-pm-evals pytest evals/tests
```
