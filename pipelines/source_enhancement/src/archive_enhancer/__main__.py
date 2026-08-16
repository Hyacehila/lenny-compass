from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .deepseek_agent import DeepSeekAgent
from .frontmatter import build_enhanced_copy
from .pipeline import chapter_all, report_state
from .provenance import ENHANCED_MANIFEST_NAME, write_enhanced_manifest
from .rebuild import rebuild_newsletters
from .validator import validate_all, write_source_snapshot


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = REPO_ROOT / "corpus" / "raw"
DEFAULT_OUTPUT = REPO_ROOT / "corpus" / "enhanced"
DEFAULT_RUNTIME = REPO_ROOT / "runtime" / "pipelines" / "source-enhancement"
DEFAULT_PROVENANCE = PIPELINE_ROOT / "provenance"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhance the private archive with a local DeepSeek Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="rebuild the complete enhanced corpus from raw Markdown")
    prepare.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    prepare.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    prepare.add_argument("--reset-runtime", action="store_true", help="clear ignored Agent state before rebuilding")

    rebuild = sub.add_parser("rebuild-newsletters", help="safely rebuild only the enhanced newsletter directory")
    rebuild.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    rebuild.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    rebuild.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    rebuild.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)

    chapter = sub.add_parser("chapter", help="ask DeepSeek to create Podcast chapter plans")
    chapter.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    chapter.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    chapter.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    chapter.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    chapter.add_argument("--model", default="deepseek-v4-flash")
    chapter.add_argument("--concurrency", type=int, default=3)
    chapter.add_argument("--max-retries", type=int, default=3)
    chapter.add_argument("--review-sample", type=int, default=10)
    chapter.add_argument("--resume", action="store_true")
    chapter.add_argument("--only", nargs="*", help="optional Podcast filenames for a pilot run")

    validate = sub.add_parser("validate", help="validate the enhanced layer")
    validate.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    validate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate.add_argument("--provenance", "--runtime", dest="provenance", type=Path, default=DEFAULT_PROVENANCE)
    validate.add_argument("--stage", choices=["prepared", "final"], default="final")

    manifest = sub.add_parser("manifest", help="write the per-document enhanced provenance manifest")
    manifest.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    manifest.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    manifest.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)

    sub.add_parser("report", help="show local Agent state and usage")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = make_parser().parse_args(argv)
    if args.command == "prepare":
        if args.reset_runtime and args.runtime.exists():
            import shutil

            shutil.rmtree(args.runtime)
        result = build_enhanced_copy(args.source, args.output)
        result["source_files"] = write_source_snapshot(args.source, args.provenance)["file_count"]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "rebuild-newsletters":
        result = rebuild_newsletters(args.source, args.output, args.runtime, args.provenance)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "chapter":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            print("DEEPSEEK_API_KEY is not set; refusing to call the API.", file=sys.stderr)
            return 2
        agent = DeepSeekAgent(model=args.model, max_retries=args.max_retries)
        result = chapter_all(
            source=args.source,
            output=args.output,
            runtime=args.runtime,
            agent=agent,
            concurrency=max(1, args.concurrency),
            review_sample=max(0, min(100, args.review_sample)),
            resume=args.resume,
            only=args.only,
        )
        if result["failed"] == 0 and args.only is None:
            validation = validate_all(
                args.source,
                args.output,
                require_chapters=True,
                runtime=args.provenance,
            )
            if not validation["valid"]:
                print(json.dumps(validation, ensure_ascii=False, indent=2), file=sys.stderr)
                return 1
            manifest = write_enhanced_manifest(args.source, args.output, args.provenance)
            result["enhanced_manifest_documents"] = manifest["document_count"]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["failed"] == 0 else 1
    if args.command == "validate":
        result = validate_all(
            args.source,
            args.output,
            require_chapters=args.stage == "final",
            runtime=args.provenance,
            require_manifest=args.stage == "final",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["valid"] else 1
    if args.command == "manifest":
        validation = validate_all(
            args.source,
            args.output,
            require_chapters=True,
            runtime=args.provenance,
        )
        if not validation["valid"]:
            print(json.dumps(validation, ensure_ascii=False, indent=2), file=sys.stderr)
            return 1
        manifest = write_enhanced_manifest(args.source, args.output, args.provenance)
        print(
            json.dumps(
                {
                    "path": str(args.provenance / ENHANCED_MANIFEST_NAME),
                    "documents": manifest["document_count"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "report":
        print(json.dumps(report_state(DEFAULT_RUNTIME), ensure_ascii=False, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
