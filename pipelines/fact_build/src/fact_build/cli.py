from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .builder import FactBuilder, load_default_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fact-build",
        description="Build the local, structure-preserving Docling fact layer.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config/facts/build.json (auto-detected from a source checkout).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the fixed-revision tokenizer to already exist in the local HF cache.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pilot", help="Run and persist the required five-document gate.")
    build = subparsers.add_parser("build", help="Run the pilot gate, then build all 638 documents.")
    build.add_argument(
        "--skip-pilot",
        action="store_true",
        help="Skip a repeated pilot only when a passing report has already been reviewed.",
    )
    subparsers.add_parser("validate", help="Validate every published fact artifact and hash.")
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUTF8", "1")
    arguments = _parser().parse_args(argv)
    config = load_default_config(arguments.config)
    builder = FactBuilder(
        config,
        progress=lambda message: print(message, flush=True),
        local_files_only=arguments.local_files_only,
    )
    if arguments.command == "pilot":
        result = builder.run_pilot()
    elif arguments.command == "build":
        manifest = builder.run_full_build(require_pilot=not arguments.skip_pilot)
        result = {
            "build_id": manifest.build_id,
            "document_count": manifest.document_count,
            "chunk_count": manifest.chunk_count,
        }
    else:
        result = builder.validate_published()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
