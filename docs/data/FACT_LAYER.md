# Fact layer contract

The fact build consumes exactly the 638 Markdown files under `corpus/enhanced/` and
side-loads authoritative metadata from matching raw front matter. The three support
files under `corpus/raw/01-start-here/` are not converted.

Versioned artifacts are organized as follows:

```text
facts/
  documents/v2/<collection>/<slug>.docling.json
  chunks/v2/<collection>/<slug>.chunks.json
  manifests/build.json
  manifests/pilot-report.json
  quarantine/
```

Paths stored in fact manifests and locators are relative to the `facts/` root.
Source Markdown paths remain repository-relative. Document IDs are the stable
collection/filename pair and never use a title as a key.

Each locator names the raw Markdown, enhanced Markdown, Docling JSON, and fact JSON
paths explicitly. The corresponding SHA-256 values are verified again from disk on
every fact read; the target chunk's canonical content hash is recomputed before it
can become evidence. A mismatch is reported as `STALE_LOCATOR`.

The pilot must cover a short heading-free newsletter, the largest complex
newsletter, fenced code, native HTML, and the largest podcast. A full build starts
automatically only when all five pass JSON round-trip, structure, image, reference,
ordinal, hash, and 1500-token chunk checks.

The full acceptance gate is 638 successful documents and zero quarantine entries.
