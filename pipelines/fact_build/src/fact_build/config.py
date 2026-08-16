from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FactBuildConfigError(ValueError):
    """Raised when the fact-build configuration is unsafe or inconsistent."""


@dataclass(frozen=True)
class FactBuildConfig:
    path: Path
    repo_root: Path
    data: dict[str, Any]
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> FactBuildConfig:
        resolved = Path(path).resolve(strict=True)
        raw = resolved.read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FactBuildConfigError(f"invalid JSON configuration: {resolved}") from exc

        repo_root = resolved.parents[2]
        config = cls(
            path=resolved,
            repo_root=repo_root,
            data=data,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        config.validate()
        return config

    @property
    def config_version(self) -> str:
        return str(self.data["config_version"])

    @property
    def max_tokens(self) -> int:
        return int(self.data["chunking"]["max_tokens"])

    @property
    def expected_document_count(self) -> int:
        return int(self.data["input"]["expected_document_count"])

    @property
    def pilot_documents(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.data["pilot"]["documents"])

    def repo_path(self, relative_path: str) -> Path:
        candidate_input = Path(relative_path)
        if candidate_input.is_absolute():
            raise FactBuildConfigError(f"absolute repository path is forbidden: {relative_path}")
        candidate = (self.repo_root / candidate_input).resolve(strict=False)
        try:
            candidate.relative_to(self.repo_root)
        except ValueError as exc:
            raise FactBuildConfigError(
                f"configured path escapes repository root: {relative_path}"
            ) from exc
        return candidate

    def input_path(self, key: str) -> Path:
        return self.repo_path(str(self.data["input"][key]))

    def output_path(self, key: str) -> Path:
        return self.repo_path(str(self.data["output"][key]))

    def facts_relative(self, repository_path: Path) -> str:
        facts_root = (self.repo_root / "facts").resolve(strict=False)
        try:
            return repository_path.resolve(strict=False).relative_to(facts_root).as_posix()
        except ValueError as exc:
            raise FactBuildConfigError(
                f"fact artifact is not beneath facts root: {repository_path}"
            ) from exc

    def validate(self) -> None:
        if self.data.get("schema_version") != 1:
            raise FactBuildConfigError("schema_version must be 1")
        if not self.data.get("config_version"):
            raise FactBuildConfigError("config_version must be non-empty")

        docling = self.data.get("docling", {})
        expected_docling = {
            "version": "2.120.1",
            "core_version": "2.91.0",
            "input_format": "markdown",
            "pipeline": "SimplePipeline",
            "fetch_remote_images": False,
        }
        for key, value in expected_docling.items():
            if docling.get(key) != value:
                raise FactBuildConfigError(f"docling.{key} must be {value!r}")

        chunking = self.data.get("chunking", {})
        expected_chunking = {
            "chunker": "HybridChunker",
            "tokenizer_model": "sentence-transformers/all-MiniLM-L6-v2",
            "tokenizer_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "max_tokens": 1500,
            "merge_peers": True,
        }
        for key, value in expected_chunking.items():
            if chunking.get(key) != value:
                raise FactBuildConfigError(f"chunking.{key} must be {value!r}")

        pilot = self.pilot_documents
        if len(pilot) != 5 or len(set(pilot)) != 5:
            raise FactBuildConfigError("pilot.documents must contain exactly five unique paths")
        required_pilot = {
            "02-newsletters/get-perplexity-pro-free-for-1-year.md",
            "02-newsletters/how-to-kickstart-and-scale-a-consumer-business.md",
            "02-newsletters/how-to-measure-cohort-retention.md",
            "02-newsletters/how-to-run-seo-experiments.md",
            "03-podcasts/eric-ries.md",
        }
        if set(pilot) != required_pilot:
            raise FactBuildConfigError("pilot.documents does not match the required five-document gate")

        output_keys = (
            "documents_root",
            "chunks_root",
            "assets_root",
            "quarantine_root",
            "build_manifest",
            "pilot_report",
        )
        for key in output_keys:
            path = self.output_path(key)
            try:
                path.relative_to((self.repo_root / "facts").resolve(strict=False))
            except ValueError as exc:
                raise FactBuildConfigError(f"output.{key} must be beneath facts/") from exc
