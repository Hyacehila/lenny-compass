# Local retrieval builder

This Python 3.11 workspace package builds two independent, disposable SQLite
indexes from `facts/manifests/build.json`. The CLI's lexical and semantic output
arguments describe their stable layout inside each immutable generation:

- `retrieval/lexical/fts.sqlite3` uses FTS5/BM25 over title, headings, metadata,
  and body with weights `5/3/2/1`;
- `retrieval/semantic/vec.sqlite3` uses sqlite-vec 0.1.9 cosine search over
  2048-dimensional Zhipu `embedding-3` vectors.

Each build is installed beneath `retrieval/generations/<build-id>/`. The bundle
manifest is the single pointer published last at `retrieval/manifests/build.json`;
it binds both generation database paths to the exact fact manifest hash and build
ID. Legacy manifests that reference the original fixed database paths remain
readable. When building from the legacy v2 fact layout, the builder materializes
a one-time immutable copy of that fact manifest beneath `facts/generations/`.
Generated retrieval artifacts are local state and are not committed.

Runtime consumers can call `load_and_resolve_bundle_manifest(...)` to obtain the
validated absolute database and fact-manifest paths. Pass the returned
`facts_manifest_path` to `FactStore(facts_root, manifest_path=...)` so evidence
reads remain bound to the same snapshot after the active facts pointer advances.

Build lexical retrieval without publishing a placeholder semantic database:

```powershell
uv run --package retrieval-build retrieval-build build `
  --facts-manifest facts/manifests/build.json `
  --semantic none
```

Build both databases after placing `ZHIPU_API_KEY` in the ignored root `.env`:

```powershell
uv run --env-file .env --package retrieval-build retrieval-build build `
  --facts-manifest facts/manifests/build.json `
  --semantic zhipu --require-semantic
```

Embedding inputs are normally one-to-one with HybridChunker facts. A deterministic
secondary split occurs only for Zhipu's explicit HTTP 400 / business code 1261
input-length error. Its segment ordinal is stored in the semantic database while
the canonical parent locator remains unchanged for fact resolution.

```powershell
uv run --package retrieval-build retrieval-build inspect `
  --manifest retrieval/manifests/build.json
uv run --package retrieval-build pytest pipelines/retrieval_build/tests
```
