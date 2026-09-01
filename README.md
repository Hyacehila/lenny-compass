# Lenny Compass

Private, local-first RAG workspace for a paid Lenny's Newsletter and Podcast
archive. This repository must remain private: it contains subscriber-only source
material and derived full-text facts.

## What is in the repository

- `corpus/raw/` is the immutable 641-file upstream snapshot. Do not normalize or
  edit it. The knowledge corpus is 349 newsletters plus 289 podcast transcripts.
- `corpus/enhanced/` contains the 638 structure-preserving Markdown documents.
  Raw frontmatter remains the metadata authority.
- `facts/` contains the versioned local Docling fact layer: 638 lossless
  `DoclingDocument` JSON files and 638 HybridChunker artifacts with 11,286 chunks.
- `retrieval/` contains versioned local SQLite indexes. The two generated databases
  use Git LFS; FTS5 is usable without a model key and sqlite-vec requires Zhipu
  `embedding-3`.
- `backend/` is the localhost FastAPI/SSE service, and `frontend/` is the React
  interface it serves after a production build.
- `contracts/` contains the strict fact, locator, retrieval, evidence, and SSE
  schemas shared by the pipelines and backend.

The retrieval databases locate facts but are never answer evidence. Both Direct
and Agentic modes must resolve a locator through the fact layer before the text can
reach an answer model. See `docs/architecture/RAG_SYSTEM.md` and
`docs/decisions/0001-facts-before-answers.md`. Phase 1 verifies exact citation
provenance and immutable-snapshot reads. Stage 2 adds an independent model call
that checks every atomic claim against the exact Evidence it cites, allows one
repair attempt, and publishes only claims that pass. This is an automatic
evidence-support check rather than a claim that the outside world has been
fact-checked. The Phase 1 boundary is recorded in
`docs/decisions/0002-structural-grounding-boundary.md`.

The upstream license is preserved at `corpus/raw/01-start-here/LICENSE.md`. Read
`NOTICE.md`, `docs/PROVENANCE.md`, and `docs/USAGE_POLICY.md` before using the data.

## Local setup

The Python workspace is locked to Python 3.11 and managed with uv. The web
workspace uses pnpm.

```powershell
uv sync --all-packages --frozen
pnpm install --frozen-lockfile
```

Create a repository-local `.env` only when model access is needed. It is ignored by
Git and stores the real values in plain text on this machine:

```powershell
Copy-Item .env.example .env
```

`DEEPSEEK_API_KEY` enables question planning, answer writing, the independent
claim verifier, and both user-selectable answer modes. `DEEPSEEK_VERIFY_MODEL`
may point to a separate verifier model; by default it uses the same supported
DeepSeek model as the writer.
`ZHIPU_API_KEY` enables the 2048-dimensional vector index and optional image
inspection. The template fixes the supported model names and endpoints; only the
two key values remain local. Never put a real key in `.env.example`, source code, a
manifest, or a commit.

## Validate or rebuild facts

The checked-in fact layer was produced by Docling's Markdown `SimplePipeline`, with
remote image fetching disabled, followed by HybridChunker using a pinned MiniLM
tokenizer revision and a 1,500-token maximum. It retains image URLs and captions,
not image bytes.

```powershell
uv run --package lenny-compass-fact-build fact-build validate
```

An intentional rebuild always runs the five-document pilot before the full corpus:

```powershell
uv run --package lenny-compass-fact-build fact-build pilot
uv run --package lenny-compass-fact-build fact-build build
uv run --package lenny-compass-fact-build fact-build validate
```

## Build retrieval indexes

Build the production FTS5 database without a key. This explicitly records the
semantic index as not ready and does not create fake vectors:

```powershell
uv run --package retrieval-build retrieval-build build `
  --facts-manifest facts/manifests/build.json `
  --semantic none
```

After adding a Zhipu key to `.env`, build both databases and require the semantic
artifact to succeed:

```powershell
uv run --env-file .env --package retrieval-build retrieval-build build `
  --facts-manifest facts/manifests/build.json `
  --semantic zhipu --require-semantic
```

The generated bundle manifest is `retrieval/manifests/build.json`; the independent
databases are `retrieval/lexical/fts.sqlite3` and
`retrieval/semantic/vec.sqlite3`.

## Run the application

```powershell
pnpm web:build
uv run --package lenny-backend lenny-backend
```

Open `http://127.0.0.1:4317`. Conversations are stored in ignored local SQLite
state under `runtime/`. The interface calls the two modes “快速模式” and “深入模式”:
the first performs one retrieval pass, while the second lets an agent search and
read source windows over multiple rounds. Both run the independent claim verifier,
so verification adds latency and another model call. Open
`http://127.0.0.1:4317/#/guide` for the in-product usage guide.

For scenario questions, Compass may ask one decision-relevant question at a time,
up to three times. The user may answer, choose a suggested reply, or skip. Answers
separate local-corpus findings, user-provided premises, evidence-derived advice,
and general PM guidance. General guidance is allowed when local evidence is absent,
but it is explicitly labeled and never presented as a corpus finding.

Stage 3 is intentionally out of scope. There is no web search, file upload,
automatic workflow generation, collaboration account system, or LLM Wiki retrieval
in the current product.

## Test the layers

```powershell
uv run --package lenny-compass-fact-build pytest pipelines/fact_build/tests
uv run --package retrieval-build pytest pipelines/retrieval_build/tests
uv run --package lenny-backend pytest backend/tests
pnpm web:test
pnpm web:build
```

Validate the original migration separately because that package deliberately has
its own lock and environment:

```powershell
Push-Location pipelines/source_enhancement
$env:PYTHONPATH = (Resolve-Path "src").Path
uv sync --frozen
uv run python -m archive_enhancer validate --stage final
uv run python -m unittest discover -s tests -v
Pop-Location
```

## Retired migration-era paths and tools

The old taxonomy, reading paths, JavaScript archive validators, and `apps/` /
`packages/core/` placeholders remain retired; their last versions are in commit
`c360399`. The current root pnpm workspace is a new React application, not a
restoration of those archive validators. Corpus validation is owned by the Python
source-enhancement pipeline and its per-file provenance manifest.
