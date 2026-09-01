# 0002: Structural grounding is not semantic entailment

## Status

Accepted for phase 1, with a required phase-2 follow-up.

## Decision

Phase 1 uses a fail-closed, structural grounding boundary. Every visible grounded
answer line must cite an exact Evidence ID that was resolved from the immutable
facts snapshot in the same run. Only actually cited Evidence and derived claims are
persisted. An answer that cannot be supported must use the deterministic
`insufficient_evidence` response with no persisted Evidence or claims.

This proves citation provenance and run-time snapshot integrity. It does **not**
prove that the prose of a claim is semantically entailed by the cited passage. A
model can still attach a valid, read Evidence ID to a claim that the passage does
not support. The deterministic evaluation package likewise measures structural
claim-to-evidence mappings, not semantic factuality.

Product copy and reports must describe phase-1 answers as source-linked or
evidence-cited, not independently fact-checked. A future semantic gate must be
fail-closed and evaluated against human-labeled claim/evidence pairs before this
boundary can be described as entailment verification.

## Consequences

- Invalid, unread, stale, or missing citations fail the run.
- Retrieval noise may end in a successful deterministic refusal even after source
  windows were read.
- Snapshot changes abort the whole run; partial evidence from mixed generations is
  never accepted.
- Semantic support remains an explicit known risk rather than a hidden product
  promise.
