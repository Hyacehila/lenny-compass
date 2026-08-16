# ADR 0001: Resolve facts before answering

Status: accepted

## Decision

Retrieval results contain rank information and a stable locator, but no citable
source excerpt. The answer path must resolve locators against the checked-in fact
artifacts and may cite only evidence produced by that resolver.

LLM Wiki is the sole exception because its generated Wiki page is itself the
abstract fact. It uses a distinct locator and evidence kind so it cannot be mistaken
for a Docling source chunk.

## Consequences

- FTS and vector databases remain safely reproducible and disposable.
- Stale indexes fail closed when source or build hashes change.
- Agentic search alone cannot authorize a citation; a read operation is required.
- Index snippets, BM25 scores, vector distances, query rewrites, and model reasoning
  are excluded from the evidence prompt.
