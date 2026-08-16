# Docling fact builder

This Python 3.11 workspace package converts the 638 enhanced Markdown bodies into
versioned Docling JSON documents and HybridChunker fact documents. Raw Markdown
frontmatter remains the metadata authority. Remote and local image fetching are
disabled; source image URLs and captions are retained as references without
downloading image bytes.

The build is pinned to `docling-slim[format-markdown,feat-chunking]==2.120.1`,
`docling-core==2.91.0`, and the immutable
`sentence-transformers/all-MiniLM-L6-v2` tokenizer revision in
`config/facts/build.json`. Docling 2.120.1 imports PDF and reading-order modules
while initializing `DocumentConverter` even for a Markdown-only converter, so
the package also pins the minimal compatible import-time dependencies explicitly.

From the repository root:

```powershell
uv sync --package lenny-compass-fact-build
uv run --package lenny-compass-fact-build fact-build pilot
uv run --package lenny-compass-fact-build fact-build build
uv run --package lenny-compass-fact-build fact-build validate
```

`build` refuses to start the full corpus unless all five pilot documents pass.
It atomically publishes `facts/documents/v2`, `facts/chunks/v2`, and
`facts/manifests/build.json`. Locator artifact paths are relative to `facts/`;
raw and enhanced paths are explicit repository-relative fields with independent
SHA-256 hashes. Each manifest entry also hashes the final fact-document bytes,
and each locator hashes its canonical chunk content. A non-empty
`facts/quarantine` makes final validation fail.
