from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from archive_enhancer.frontmatter import (
    document_heading,
    normalize_newsletter_body,
    render_enhanced_document,
    split_front_matter,
)
from archive_enhancer.validator import validate_document


def raw_newsletter(body: bytes) -> bytes:
    return (
        b"---\n"
        b'title: "Test newsletter"\n'
        b'date: "2026-08-15"\n'
        b'type: "newsletter"\n'
        b'tags: ["test"]\n'
        b"---\n"
        + body
    )


class NewsletterNormalizationTests(unittest.TestCase):
    def test_demotes_every_heading_when_body_contains_h1(self) -> None:
        body = b"# Section\n\n## Detail\n\n#### Leaf\n"

        normalized, stats = normalize_newsletter_body(body, "test.md")

        self.assertEqual(normalized, b"## Section\n\n### Detail\n\n##### Leaf\n")
        self.assertTrue(stats.normalized)
        self.assertEqual(stats.demoted_headings, 3)
        self.assertEqual(stats.h6_to_bold, 0)

    def test_demotes_blockquote_headings(self) -> None:
        body = b"> # Question\n\n## Answer\n"

        normalized, stats = normalize_newsletter_body(body, "quoted.md")

        self.assertEqual(normalized, b"> ## Question\n\n### Answer\n")
        self.assertEqual(stats.demoted_headings, 2)

    def test_ignores_heading_markers_inside_fenced_code(self) -> None:
        body = b"# Section\n\n```markdown\n# Example only\n```\n\n## Detail\n"

        normalized, stats = normalize_newsletter_body(body, "code.md")

        self.assertEqual(
            normalized,
            b"## Section\n\n```markdown\n# Example only\n```\n\n### Detail\n",
        )
        self.assertEqual(stats.demoted_headings, 2)

    def test_converts_h6_to_bold_paragraph(self) -> None:
        body = b"# Section\n\n> ###### Source: [Example](https://example.com)\n"

        normalized, stats = normalize_newsletter_body(body, "caption.md")

        self.assertEqual(
            normalized,
            b"## Section\n\n> **Source: [Example](https://example.com)**\n",
        )
        self.assertEqual(stats.demoted_headings, 1)
        self.assertEqual(stats.h6_to_bold, 1)

    def test_leaves_body_without_h1_byte_for_byte_unchanged(self) -> None:
        body = b"## Existing section\r\n\r\nText\r\n"

        normalized, stats = normalize_newsletter_body(body, "unchanged.md")

        self.assertIs(normalized, body)
        self.assertFalse(stats.normalized)

    def test_rejects_setext_heading_in_affected_body(self) -> None:
        body = b"Title\n=====\n"

        with self.assertRaisesRegex(ValueError, "unsupported non-ATX heading"):
            normalize_newsletter_body(body, "setext.md")

    def test_validator_rejects_blockquoted_second_h1(self) -> None:
        source_data = raw_newsletter(b"> # Question\n\nAnswer\n")
        fields, body = split_front_matter(source_data)
        incorrect = document_heading("newsletter", fields).encode("utf-8") + b"\n" + body

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.md"
            enhanced = root / "enhanced.md"
            source.write_bytes(source_data)
            enhanced.write_bytes(incorrect)

            result = validate_document(source, enhanced, require_chapters=False)

        self.assertFalse(result["valid"])
        self.assertIn("expected exactly one semantic H1, found 2", result["errors"])

    def test_rendered_newsletter_passes_validator(self) -> None:
        source_data = raw_newsletter(b"# Section\n\n## Detail\n")
        fields, body = split_front_matter(source_data)
        rendered, stats = render_enhanced_document("newsletter", fields, body, "source.md")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.md"
            enhanced = root / "enhanced.md"
            source.write_bytes(source_data)
            enhanced.write_bytes(rendered)

            result = validate_document(source, enhanced, require_chapters=False)

        self.assertTrue(stats.normalized)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["h1"], 1)


if __name__ == "__main__":
    unittest.main()
