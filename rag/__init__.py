# rag/__init__.py

from .indices import (
    BM25Index,
    DenseIndex,
    build_bm25_index,
    build_dense_index,
)

from .retrieval import (
    bm25_search,
    dense_search,
    hybrid_retrieve,
)

from .rerank import (
    rerank_cross_encoder,
    apply_mmr,
)

from .context import (
    build_context_window,
)

from .corpus import (
    Document,
    Chunk,
    Corpus,
    build_corpus_from_texts,
    add_chunks_to_corpus,
    get_chunk_texts,
    get_chunk,
    get_document,
)

# Опционально: токены/дебаг, если решишь их делать
# from .tokens import count_tokens_simple, truncate_text_to_tokens
# from .debug import inspect_retrieval, log_retrieval_stats

__all__ = [
    "BM25Index",
    "DenseIndex",
    "build_bm25_index",
    "build_dense_index",
    "bm25_search",
    "dense_search",
    "hybrid_retrieve",
    "rerank_cross_encoder",
    "apply_mmr",
    "build_context_window",
    "Document",
    "Chunk",
    "Corpus",
    "build_corpus_from_texts",
    "add_chunks_to_corpus",
    "get_chunk_texts",
    "get_chunk",
    "get_document",
    # "count_tokens_simple",
    # "truncate_text_to_tokens",
    # "inspect_retrieval",
    # "log_retrieval_stats",
]
