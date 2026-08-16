from __future__ import annotations

import re
from collections.abc import Iterable

from ai_pm_contracts import QueryRewrite

_FALLBACK_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.\-]{1,}")
_STOP_WORDS = {
    "about",
    "and",
    "are",
    "did",
    "does",
    "for",
    "from",
    "how",
    "into",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _quote_fts_phrase(value: str) -> str:
    normalized = " ".join(value.split())
    return f'"{normalized.replace(chr(34), chr(34) * 2)}"'


def _unique_nonempty(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if normalized and key not in seen:
            result.append(normalized)
            seen.add(key)
    return result


def compile_fts_match(rewrite: QueryRewrite) -> str | None:
    """Compile validated terms to bound FTS5 syntax without accepting raw SQL."""

    values = _unique_nonempty((*rewrite.lexical_phrases_en, *rewrite.lexical_terms_en))
    if not values:
        values = [
            token
            for token in _unique_nonempty(
                _FALLBACK_TOKEN.findall(rewrite.semantic_query_en)
            )
            if token.casefold() not in _STOP_WORDS
        ][:8]
    if not values:
        return None
    return " OR ".join(_quote_fts_phrase(value) for value in values)
