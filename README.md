# Lenny Compass

Private, local-first workspace for a paid Lenny's Newsletter and Podcast archive.
This repository must remain private. It contains subscriber-only source material and
derived full-text copies.

## Data layers

- `corpus/raw/` is the immutable upstream snapshot. Its 641 files were moved from
  the former `archive/` directory without byte changes. Do not edit or normalize
  files in this directory.
- `corpus/enhanced/` contains 638 structure-preserving Markdown documents derived
  from the 349 newsletters and 289 podcast transcripts. Newsletter title conflicts
  are normalized deterministically; podcast H2/H3 chapter headings were generated
  by the source-enhancement pipeline. Raw front matter remains the metadata authority.
- `pipelines/source_enhancement/` contains the Python 3.11 transformation,
  validation, tests, and provenance records for the enhanced layer.
- `facts/` is reserved for future knowledge-layer artifacts. It is not a source of
  truth for this migration.
- `retrieval/` and `runtime/` hold generated local indexes and application state and
  are ignored by Git except for directory placeholders.
- `backend/`, `frontend/`, `contracts/`, and `evals/` are application scaffolding;
  they are not part of the corpus migration.

The upstream license is preserved at
`corpus/raw/01-start-here/LICENSE.md`. See `NOTICE.md`,
`docs/PROVENANCE.md`, and `docs/USAGE_POLICY.md` before using the data.

## Validate the migration

The validation command does not require an API key. From PowerShell:

```powershell
Push-Location pipelines/source_enhancement
$env:PYTHONPATH = (Resolve-Path "src").Path
uv sync --frozen
uv run python -m archive_enhancer validate --stage final
uv run python -m unittest discover -s tests -v
Pop-Location
```

Final validation checks the raw SHA-256 snapshot, the upstream index against raw
front matter, the raw/enhanced document mapping, structural and body fidelity, and
the checked-in per-document enhanced manifest. Regenerate that manifest only after
an intentional, validated enhanced-layer rebuild:

```powershell
Push-Location pipelines/source_enhancement
$env:PYTHONPATH = (Resolve-Path "src").Path
uv run python -m archive_enhancer manifest
Pop-Location
```

## Secrets

Copy `.env.example` to an ignored `.env` only when a pipeline actually needs an API
credential. Never place a real key in `.env.example`, source code, provenance JSON,
or any committed file. Corpus and provenance validation are fully local.

## Retired migration-era paths and tools

The following files from the initial repository layout are intentionally retired;
their last historical versions remain in Git commit `c360399`:

- `config/taxonomy.yml` and `config/reading-paths.yml`;
- the root `package.json`, `pnpm-lock.yaml`, and `pnpm-workspace.yaml`;
- `scripts/generate-archive-manifest.mjs` and `scripts/validate-archive.mjs`;
- the old `apps/` and `packages/core/` placeholders.

The Python source-enhancement validator and the manifests under
`pipelines/source_enhancement/provenance/` replace the old path-specific JavaScript
archive checks. Do not restore the retired tools without updating them for the
current `corpus/raw` and `corpus/enhanced` contract.
