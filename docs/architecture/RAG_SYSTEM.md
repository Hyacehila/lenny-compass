# Local RAG architecture

Lenny Compass is a localhost-only retrieval and answering system. It keeps the
versioned fact layer separate from disposable retrieval indexes: indexes locate a
fact, while only a resolved fact may be given to an answer model.

## Data flow

1. `corpus/raw/` remains the immutable metadata and source snapshot.
2. `corpus/enhanced/` supplies structure-preserving Markdown to Docling's
   `SimplePipeline`.
3. `pipelines/fact_build/` stores one Docling JSON document and one validated chunk
   artifact per source under `facts/`.
4. `pipelines/retrieval_build/` builds independent FTS5 and sqlite-vec databases
   under ignored `retrieval/` paths.
5. The backend resolves every retrieval locator through `FactStore` before an
   answer is generated.

Direct mode rewrites the question once, retrieves up to 50 keyword and 50 semantic
candidates, combines ranks with equal-weight RRF (`k=60`), retains 12 locators, and
resolves each at `window=0`. Retrieval snippets and scores never become evidence.

The vector builder normally stores one vector per HybridChunker block. It performs
a deterministic secondary text split only when Zhipu explicitly identifies an
input-length rejection. Those vector segments keep a retrieval segment ordinal but
all resolve to the unchanged parent fact locator; unrelated provider failures are
never treated as length errors.

Agentic mode is disabled by default. When explicitly enabled, its tools return
locators rather than excerpts. The agent chooses searches and source windows, and
may cite only evidence returned by a source-read or image-inspection tool. Agentic
mode does not perform automatic RRF or a hidden rewrite.

## Source windows

A `DocumentChunkLocator` includes explicit raw/enhanced paths and hashes, Docling
self references, chunk and segment ordinals, and fact artifact paths. `window=N`
returns the located chunk plus up to N chunks before and after it in the same
document. `window=0` returns exactly one chunk. Raw, enhanced, Docling, fact, or
canonical chunk hash mismatches fail closed with `STALE_LOCATOR`.

Markdown conversion does not produce source line or byte offsets. Docling JSON
pointers and versioned chunk ordinals are therefore the canonical locations.

## Images

Fact conversion does not fetch remote images. It retains the remote URL, caption,
and Docling reference. Direct answers may ask `glm-5v-turbo` to inspect at most four
unique ranked images. The resulting description is marked as a derived visual
observation and remains tied to its source locator. If inspection fails, the answer
receives only the caption and URL plus an explicit failure marker.

## Model boundary

- DeepSeek query rewriting uses JSON output with thinking disabled.
- Final and Agentic answers use `deepseek-v4-flash` with thinking enabled and
  `reasoning_effort=max`.
- `reasoning_content` is round-tripped only inside a provider tool loop. It is not
  persisted or sent through SSE.
- Zhipu `embedding-3` is requested with 2048 dimensions; `glm-5v-turbo` is used for
  optional visual observations.

## LLM Wiki reservation

The nashsu/llm_wiki HTTP adapter is registered but disabled. It is available only to
future Agentic tools and never participates in Direct RRF. Wiki evidence uses a
stable `(project_id, path)` locator, represents an abstract fact page, and does not
support source windows. Source rescan is a control-plane operation and is never
called from an answer request.

## Runtime boundary

FastAPI binds to `127.0.0.1:4317` and serves the built React application from the
same origin. Conversations, model caches, retrieval databases, logs, and real
credentials are ignored local state. The browser receives public progress and answer
events only; it never receives provider keys or model reasoning.
