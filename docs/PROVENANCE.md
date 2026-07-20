# Archive provenance

## Snapshot

- Archive format: Lenny's Newsletter and Podcast paid-subscriber full archive.
- Snapshot root: `archive/`.
- Index source of truth: `archive/01-start-here/index.json`.
- Index schema version: `2.0`.
- Index generated at: `2026-03-19T16:44:51.213182+00:00`.
- Expected contents: 349 newsletter posts and 289 podcast transcripts.

`archive-manifest.json` records the exact file list, byte sizes, and SHA-256 checksums for this imported snapshot. Regenerate it only when deliberately replacing or extending the source snapshot.

## Update policy

The source archive is a point-in-time snapshot. New archive downloads must be imported as an intentional update, reviewed with a file-level diff, documented here, and committed separately. Never silently overwrite the current snapshot.
