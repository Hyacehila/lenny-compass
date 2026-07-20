# Lenny Compass

Private, local-first knowledge-base workspace for a paid Lenny's Newsletter and Podcast archive.

## Repository rules

- Keep this GitHub repository **private**.
- `archive/` is a byte-preserved source snapshot. Do not edit, rename, or add derived files there.
- Keep runtime data, generated SQLite databases, chat history, indexes, and secrets out of Git.
- Do not deploy this repository to GitHub Pages or any public hosting service.

## Layout

- `archive/` — original subscriber archive and upstream license.
- `docs/` — provenance, usage policy, architecture, and future public-blog outline.
- `config/` — personal taxonomy and reading-path definitions.
- `apps/api/` — reserved localhost API application.
- `apps/web/` — reserved local browser workbench.
- `packages/core/` — reserved shared parsing, retrieval, and citation logic.
- `scripts/` — archive manifest and integrity validation scripts.
- `runtime/` — ignored local databases, indexes, notes, logs, and chat history.

## Verify the source snapshot

```powershell
pnpm archive:validate
```

Regenerate the checked-in archive manifest only after intentionally importing a new source snapshot:

```powershell
pnpm archive:manifest
pnpm archive:validate
```

## Planned local application

The future application runs only on `http://127.0.0.1:4317`. It will import the archive into SQLite FTS5, provide local search and reading, and optionally use a backend-only DeepSeek API key for evidence-grounded Chinese answers. See `docs/ARCHITECTURE.md`.
