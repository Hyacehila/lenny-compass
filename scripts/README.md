# Maintenance scripts

- `generate-archive-manifest.mjs` writes the tracked file-level source snapshot manifest.
- `validate-archive.mjs` verifies the archive index, expected content counts, file paths, and manifest hashes.

Run `pnpm archive:manifest` only after a deliberate source update. Run `pnpm archive:validate` before every commit that changes `archive/` or its manifest.
