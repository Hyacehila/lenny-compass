from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from archive_enhancer.frontmatter import document_heading, split_front_matter
from archive_enhancer.provenance import (
    ENHANCED_MANIFEST_NAME,
    build_enhanced_manifest,
    metadata_values_equal,
    validate_enhanced_manifest,
    validate_index_metadata,
)
from archive_enhancer.renderer import render_podcast
from archive_enhancer.validator import validate_document


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "corpus" / "raw"
ENHANCED = REPO_ROOT / "corpus" / "enhanced"
PROVENANCE = REPO_ROOT / "pipelines" / "source_enhancement" / "provenance"


class CorpusMigrationIntegrityTests(unittest.TestCase):
    def test_raw_tree_matches_source_snapshot_byte_for_byte(self) -> None:
        snapshot = json.loads((PROVENANCE / "source-sha256.json").read_text(encoding="utf-8"))
        expected = {item["path"]: item for item in snapshot["files"]}
        actual = {
            path.relative_to(RAW).as_posix(): path
            for path in RAW.rglob("*")
            if path.is_file()
        }

        self.assertEqual(set(actual), set(expected))
        for relative, path in actual.items():
            data = path.read_bytes()
            self.assertEqual(len(data), expected[relative]["size"], relative)
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected[relative]["sha256"], relative)

    def test_index_and_frontmatter_metadata_agree_with_nbsp_comparison_only(self) -> None:
        result = validate_index_metadata(RAW)

        self.assertTrue(result["valid"], result["differences"][:3])
        self.assertEqual(result["documents"], 638)
        self.assertTrue(metadata_values_equal("guest post\u00a0by Katie", "guest post by Katie"))
        self.assertFalse(metadata_values_equal("guest  post", "guest post"))

    def test_unicode_document_identity_and_metadata_are_preserved(self) -> None:
        manifest = build_enhanced_manifest(RAW, ENHANCED, PROVENANCE)
        document_id = (
            "02-newsletters/summary-lessons-from-working-with-600-yc-startups--"
            "gustaf-alströmer-y-combinator.md"
        )
        record = next(item for item in manifest["documents"] if item["document_id"] == document_id)

        self.assertEqual(record["metadata_authority"], "raw_frontmatter")
        self.assertIn("Gustaf", record["metadata"]["title"])
        self.assertIn("alströmer", record["raw"]["path"])
        self.assertIn("alströmer", record["enhanced"]["path"])


class EnhancedManifestTests(unittest.TestCase):
    def test_checked_in_manifest_matches_every_raw_and_enhanced_file(self) -> None:
        expected = json.loads((PROVENANCE / ENHANCED_MANIFEST_NAME).read_text(encoding="utf-8"))
        actual = build_enhanced_manifest(RAW, ENHANCED, PROVENANCE)

        self.assertEqual(expected, actual)
        self.assertEqual(actual["document_count"], 638)
        self.assertEqual(len(actual["documents"]), 638)
        self.assertEqual(actual["pipeline"]["podcast_chaptering"]["model"], "deepseek-v4-flash")

    def test_manifest_validator_reports_current_corpus_as_valid(self) -> None:
        result = validate_enhanced_manifest(RAW, ENHANCED, PROVENANCE)

        self.assertTrue(result["valid"], result.get("error"))
        self.assertEqual(result["documents"], 638)


class PodcastEnhancementTests(unittest.TestCase):
    def test_unicode_podcast_render_preserves_all_turns_and_adds_chapters(self) -> None:
        raw_data = (
            "---\n"
            'title: "Unicode conversation"\n'
            'date: "2026-08-16"\n'
            'type: "podcast"\n'
            'guest: "Gustaf Alströmer"\n'
            'channel: "Lenny\'s Podcast"\n'
            'description: "A migration test."\n'
            'tags: ["test"]\n'
            "word_count: 12\n"
            "---\n"
            "\n"
            "**Lenny** (00:00:01):\n"
            "Welcome to the conversation.\n"
            "\n"
            "**Gustaf Alströmer** (00:00:08):\n"
            "Tack så mycket.\n"
        ).encode("utf-8")
        fields, body = split_front_matter(raw_data)
        plan = {
            "chapters": [
                {
                    "title": "Introductions",
                    "sections": [
                        {
                            "title": "Welcome",
                            "start_turn": 0,
                            "end_turn": 0,
                            "approx_word_count": 4,
                        },
                        {
                            "title": "Guest greeting",
                            "start_turn": 1,
                            "end_turn": 1,
                            "approx_word_count": 3,
                        },
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gustaf-alströmer.md"
            enhanced = root / "gustaf-alströmer.md.enhanced"
            source.write_bytes(raw_data)
            enhanced.write_bytes(document_heading("podcast", fields).encode("utf-8") + b"\n" + body)
            render_podcast(enhanced, plan)

            result = validate_document(source, enhanced, require_chapters=True)
            rendered = enhanced.read_text(encoding="utf-8")

        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["h1"], 1)
        self.assertEqual(result["h2"], 1)
        self.assertEqual(result["h3"], 2)
        self.assertIn("**Gustaf Alströmer**", rendered)
        self.assertIn("Tack så mycket.", rendered)


if __name__ == "__main__":
    unittest.main()
