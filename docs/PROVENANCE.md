# Corpus provenance

## Upstream snapshot

The immutable source snapshot is stored at `corpus/raw/`. It is the paid-subscriber
Lenny's Newsletter and Podcast archive whose upstream index reports schema `2.0`,
generation time `2026-03-19T16:44:51.213182+00:00`, 349 newsletters, and 289 podcast
transcripts.

The migration from the former `archive/` path was byte-preserving: all 641 files map
one-for-one to `corpus/raw/`. `.gitattributes` disables text normalization for this
tree. `pipelines/source_enhancement/provenance/source-sha256.json` is the per-file
size and SHA-256 baseline.

## Enhanced Markdown layer

`corpus/enhanced/` contains exactly 638 Markdown documents with the same relative
newsletter and podcast filenames as the raw content documents.

- YAML front matter is removed from the rendered text, but its unmodified parsed
  values are retained as authoritative metadata in the enhanced provenance manifest.
- Every document receives one semantic H1 built from raw metadata.
- Newsletter bodies remain byte-identical unless their body contained an H1. In that
  case every semantic body heading is demoted once and H6 headings are converted to
  bold paragraphs to keep a single document H1.
- Podcast transcript text and speaker-turn order are preserved. DeepSeek-generated
  H2/H3 chapter and section headings are inserted only at speaker-turn boundaries.

The completed enhancement run is recorded in `run-summary.json`. It reports raw tree
SHA-256 `43b8267d909e1ac808a6ea8efb149553844c89633cbe924650d648e12d0c030f`
and enhanced tree SHA-256
`d3bd50f3e13bf2124e89c2d0ecd351d3dd36a59c93e69bc71506c304fcbff7c1`.

## Per-document manifest

`enhanced-manifest.json` records, for each of the 638 documents:

- stable relative document ID and document type;
- the complete raw front matter, marked as the metadata authority;
- raw and enhanced paths, byte sizes, and SHA-256 hashes;
- the applied transformation;
- pipeline version and transformation-code hashes;
- podcast model, temperature, system-prompt hash, and plan-schema hash;
- hashes of the run summary and raw source snapshot.

The historical run summary did not persist a model name for each API request. The
manifest therefore records `deepseek-v4-flash` with an explicit evidence note: it
was the source-enhancement CLI default. Per-document chapter plans were not retained,
so the enhanced file hashes are the durable identity of the generated headings.

## Metadata authority and comparison

Raw YAML front matter is authoritative. The upstream `index.json` is validated
against it but never overwrites it. Two newsletter subtitles differ only by U+00A0
NO-BREAK SPACE versus U+0020 SPACE:

- `02-newsletters/the-pm--design-partnership.md`
- `02-newsletters/this-week-9-breaking-into-growth-leading-with-influence-and-not-stepping-on-toes.md`

Validation replaces U+00A0 with U+0020 for equality comparison only. Stored raw
metadata remains unchanged.

## Update policy

Treat a new upstream download, an enhanced rebuild, and any later knowledge-layer
conversion as separate reviewed changes. Never silently overwrite `corpus/raw/` or
regenerate provenance merely to make a failed validation pass.
