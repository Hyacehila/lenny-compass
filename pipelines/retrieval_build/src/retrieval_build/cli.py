from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .build import build_retrieval_bundle, load_and_validate_bundle_manifest
from .embeddings import MockEmbeddingProvider, ZhipuEmbeddingClient
from .errors import RetrievalError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval-build")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="atomically build lexical and optional semantic indexes"
    )
    build.add_argument("--facts-manifest", type=Path, required=True)
    build.add_argument("--output-root", type=Path, default=Path("retrieval"))
    build.add_argument("--lexical-output", type=Path)
    build.add_argument("--semantic-output", type=Path)
    build.add_argument("--manifest-output", type=Path)
    build.add_argument(
        "--semantic",
        choices=("auto", "zhipu", "none", "mock"),
        default="auto",
    )
    build.add_argument("--allow-test-embedder", action="store_true")
    build.add_argument("--require-semantic", action="store_true")
    build.add_argument("--zhipu-base-url")

    inspect = subparsers.add_parser(
        "inspect", help="validate and print a build manifest"
    )
    inspect.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            manifest = load_and_validate_bundle_manifest(args.manifest)
            print(manifest.model_dump_json(indent=2))
            return 0

        embedder = None
        reason = None
        has_zhipu_key = bool(os.environ.get("ZHIPU_API_KEY"))
        if args.semantic == "mock":
            if not args.allow_test_embedder:
                raise RetrievalError(
                    "TEST_EMBEDDER_NOT_ALLOWED",
                    "--semantic mock requires --allow-test-embedder",
                )
            embedder = MockEmbeddingProvider()
            reason = "explicit test-only mock embeddings"
        elif args.semantic in {"auto", "zhipu"} and has_zhipu_key:
            embedder = ZhipuEmbeddingClient(base_url=args.zhipu_base_url)
        elif args.semantic in {"auto", "zhipu"}:
            reason = "ZHIPU_API_KEY is not set; semantic index was not built"
        else:
            reason = "semantic index explicitly disabled"

        if args.require_semantic and embedder is None:
            raise RetrievalError(
                "SEMANTIC_INDEX_REQUIRED", reason or "semantic index unavailable"
            )
        output_root = args.output_root.resolve(strict=False)
        lexical_output = args.lexical_output or output_root / "lexical" / "fts.sqlite3"
        semantic_output = (
            args.semantic_output or output_root / "semantic" / "vec.sqlite3"
        )
        manifest_output = (
            args.manifest_output or output_root / "manifests" / "build.json"
        )
        manifest = build_retrieval_bundle(
            facts_manifest_path=args.facts_manifest,
            lexical_output_path=lexical_output,
            semantic_output_path=semantic_output,
            manifest_output_path=manifest_output,
            embedder=embedder,
            semantic_reason=reason,
        )
        print(manifest.model_dump_json(indent=2))
        return 0
    except (RetrievalError, OSError, ValueError) as exc:
        code = getattr(exc, "code", type(exc).__name__)
        print(
            json.dumps(
                {"error": {"code": code, "message": str(exc)}}, ensure_ascii=False
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
