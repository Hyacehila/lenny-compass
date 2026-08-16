# Local retrieval builder

This Python 3.11 workspace package builds two independent, disposable SQLite
indexes from `facts/manifests/build.json`:

- `retrieval/lexical/fts.sqlite3` uses FTS5/BM25 over title, headings, metadata,
  and body with weights `5/3/2/1`;
- `retrieval/semantic/vec.sqlite3` uses sqlite-vec 0.1.9 cosine search over
  2048-dimensional Zhipu `embedding-3` vectors.

The bundle manifest is published last at `retrieval/manifests/build.json` and binds
both databases to the exact fact manifest hash and build ID. Generated retrieval
artifacts are local state and are not committed.

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
