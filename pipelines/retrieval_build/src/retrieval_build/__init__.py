"""Local lexical/vector retrieval index builder and fact resolver."""

from .build import (
    ValidatedRetrievalBundle,
    build_retrieval_bundle,
    load_and_resolve_bundle_manifest,
    load_and_validate_bundle_manifest,
)
from .database import (
    connect_lexical,
    connect_semantic,
    get_index_status,
    keyword_search,
    vector_search,
)
from .embeddings import EmbeddingInputTooLongError
from .fact_store import FactStore, FactStoreError
from .fusion import reciprocal_rank_fusion
from .query import compile_fts_match

__all__ = [
    "EmbeddingInputTooLongError",
    "FactStore",
    "FactStoreError",
    "ValidatedRetrievalBundle",
    "build_retrieval_bundle",
    "compile_fts_match",
    "connect_lexical",
    "connect_semantic",
    "get_index_status",
    "keyword_search",
    "load_and_resolve_bundle_manifest",
    "load_and_validate_bundle_manifest",
    "reciprocal_rank_fusion",
    "vector_search",
]
