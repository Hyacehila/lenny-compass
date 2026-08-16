from __future__ import annotations

import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from markdown_it import MarkdownIt


FRONT_MATTER_RE = re.compile(rb"\A---\r?\n(?P<header>.*?)\r?\n---\r?\n", re.S)
ATX_HEADING_RE = re.compile(
    r"^(?P<prefix>(?:(?: {0,3}>[ \t]?)+)? {0,3})"
    r"(?P<marks>#{1,6})(?P<spacing>[ \t]+)(?P<content>.*?)"
    r"(?P<ending>\r?\n?)$"
)
TRAILING_MARKS_RE = re.compile(r"[ \t]+#+[ \t]*$")
MARKDOWN = MarkdownIt("commonmark")


@dataclass
class NewsletterNormalization:
    normalized: bool = False
    demoted_headings: int = 0
    h6_to_bold: int = 0


@dataclass
class NewsletterBuildResult:
    newsletters: int = 0
    normalized_newsletters: int = 0
    demoted_headings: int = 0
    h6_to_bold: int = 0
    normalized_files: list[str] = field(default_factory=list)

    def add(self, file_name: str, normalization: NewsletterNormalization) -> None:
        self.newsletters += 1
        if normalization.normalized:
            self.normalized_newsletters += 1
            self.normalized_files.append(file_name)
        self.demoted_headings += normalization.demoted_headings
        self.h6_to_bold += normalization.h6_to_bold

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_front_matter(data: bytes) -> tuple[dict[str, Any], bytes]:
    match = FRONT_MATTER_RE.match(data)
    if not match:
        raise ValueError("Markdown file does not begin with YAML Front Matter")
    fields = yaml.safe_load(match.group("header").decode("utf-8")) or {}
    if not isinstance(fields, dict):
        raise ValueError("Front Matter must decode to a mapping")
    return fields, data[match.end():]


def _tags(fields: dict[str, Any]) -> str:
    value = fields.get("tags") or []
    if not isinstance(value, list):
        value = [value]
    return ", ".join(str(item) for item in value)


def document_heading(kind: str, fields: dict[str, Any]) -> str:
    title = str(fields.get("title") or "").strip()
    date = str(fields.get("date") or "").strip()
    tags = _tags(fields)
    if not title:
        raise ValueError("Front Matter title is empty")
    pieces = [title]
    if kind == "newsletter":
        subtitle = str(fields.get("subtitle") or "").strip()
        if subtitle:
            pieces.append(f"— {subtitle}")
    elif kind == "podcast":
        guest = str(fields.get("guest") or "").strip()
        if guest and guest.casefold() not in title.casefold():
            pieces.append(f"— {guest}")
    else:
        raise ValueError(f"Unsupported document type: {kind}")
    if date:
        pieces.append(f"| {date}")
    if tags:
        pieces.append(f"| Tags: {tags}")
    return "# " + " ".join(pieces)


def markdown_headings(text: str) -> list[tuple[int, int | None]]:
    return [
        (int(token.tag[1:]), token.map[0] if token.map is not None else None)
        for token in MARKDOWN.parse(text)
        if token.type == "heading_open"
    ]


def markdown_heading_levels(text: str) -> list[int]:
    return [level for level, _ in markdown_headings(text)]


def normalize_newsletter_body(body: bytes, source_name: str = "<newsletter>") -> tuple[bytes, NewsletterNormalization]:
    text = body.decode("utf-8")
    heading_tokens = [token for token in MARKDOWN.parse(text) if token.type == "heading_open"]
    if not any(token.tag == "h1" for token in heading_tokens):
        return body, NewsletterNormalization()

    lines = text.splitlines(keepends=True)
    result = NewsletterNormalization(normalized=True)
    for token in heading_tokens:
        if token.map is None:
            raise ValueError(f"{source_name}: Markdown heading has no source line mapping")
        line_number = token.map[0]
        match = ATX_HEADING_RE.match(lines[line_number])
        if match is None:
            raise ValueError(
                f"{source_name}:{line_number + 1}: unsupported non-ATX heading; "
                "newsletter normalization only supports ATX headings"
            )
        level = int(token.tag[1:])
        if len(match.group("marks")) != level:
            raise ValueError(f"{source_name}:{line_number + 1}: parsed heading level does not match source markers")
        if level < 6:
            lines[line_number] = (
                match.group("prefix")
                + match.group("marks")
                + "#"
                + match.group("spacing")
                + match.group("content")
                + match.group("ending")
            )
            result.demoted_headings += 1
            continue

        content = TRAILING_MARKS_RE.sub("", match.group("content")).strip()
        lines[line_number] = match.group("prefix") + f"**{content}**" + match.group("ending")
        result.h6_to_bold += 1

    normalized = "".join(lines).encode("utf-8")
    levels = markdown_heading_levels(normalized.decode("utf-8"))
    if 1 in levels:
        raise ValueError(f"{source_name}: newsletter body still contains an H1 after normalization")
    return normalized, result


def render_enhanced_document(kind: str, fields: dict[str, Any], body: bytes, source_name: str) -> tuple[bytes, NewsletterNormalization]:
    normalization = NewsletterNormalization()
    if kind == "newsletter":
        body, normalization = normalize_newsletter_body(body, source_name)
    heading = document_heading(kind, fields).encode("utf-8")
    return heading + b"\n" + body, normalization


def build_newsletter_copy(source: Path, output: Path) -> dict[str, Any]:
    src_dir = source / "02-newsletters"
    dst_dir = output / "02-newsletters"
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    result = NewsletterBuildResult()
    for src in sorted(src_dir.glob("*.md")):
        fields, body = split_front_matter(src.read_bytes())
        if fields.get("type") != "newsletter":
            raise ValueError(f"{src}: expected type='newsletter', found {fields.get('type')!r}")
        rendered, normalization = render_enhanced_document("newsletter", fields, body, src.name)
        (dst_dir / src.name).write_bytes(rendered)
        result.add(src.name, normalization)
    return result.as_dict()


def build_enhanced_copy(source: Path, output: Path) -> dict[str, Any]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        shutil.rmtree(output)
    result = build_newsletter_copy(source, output)
    src_dir = source / "03-podcasts"
    dst_dir = output / "03-podcasts"
    dst_dir.mkdir(parents=True, exist_ok=True)
    podcasts = 0
    for src in sorted(src_dir.glob("*.md")):
        fields, body = split_front_matter(src.read_bytes())
        if fields.get("type") != "podcast":
            raise ValueError(f"{src}: expected type='podcast', found {fields.get('type')!r}")
        rendered, _ = render_enhanced_document("podcast", fields, body, src.name)
        (dst_dir / src.name).write_bytes(rendered)
        podcasts += 1
    result["podcasts"] = podcasts
    return result
