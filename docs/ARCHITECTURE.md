# Planned local architecture

## Runtime boundary

- API binds only to `127.0.0.1:4317`.
- The browser workbench communicates only with that local API.
- Source Markdown remains in `archive/`; all derived state lives under ignored `runtime/` paths.

## Import and retrieval

1. Parse `archive/01-start-here/index.json` and every Markdown front matter block.
2. Store document metadata in SQLite.
3. Split newsletters at H2/H3 boundaries into chunks of at most 3,000 characters with a 200-character overlap.
4. Split podcasts on timestamped speaker/topic boundaries using the same size cap; retain speaker and start/end timestamps.
5. Store chunks in SQLite FTS5 and retain document path, heading, and timestamp references.

## Future localhost API

- `POST /api/library/import` — import or refresh a deliberately selected archive snapshot.
- `GET /api/search` — full-text search with topic, type, stage, and role filters.
- `GET /api/documents/:id` — document metadata and local source location.
- `GET /api/documents/:id/chunks` — cited local reading segments.
- `PUT /api/notes/:documentId` — save local notes and bookmarks.
- `POST /api/ask` — retrieve evidence, then return a Chinese synthesis with citations.

## DeepSeek answer pipeline

The backend alone reads `DEEPSEEK_API_KEY`. For Chinese questions it first asks the configured query model for English retrieval terms, searches local FTS5, sends only the highest-scoring evidence chunks to the answer model, and returns an answer with source citations. When evidence is insufficient, the answer must say so rather than infer unsupported claims.

Do not expose the API key, the raw archive, or local SQLite files to the web client or a public build.
