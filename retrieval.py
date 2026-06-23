from __future__ import annotations

from pathlib import Path

from logic import config, keyword_search, vector_store
from logic.embeddings import embed_query
from logic.vector_store import RetrievedChunk


def _rrf_fuse(ranked_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: birden çok sıralı listeyi tek listede birleştirir."""
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            key = chunk.text
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            chunks.setdefault(key, chunk)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [chunks[key] for key in ordered]


def get_candidates(
    question: str,
    persist_dir: Path,
    document_filter: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Aday pasajları toplar: vektör araması + (açıksa) BM25, RRF ile birleşik."""
    vector_hits = vector_store.query(
        persist_dir=persist_dir,
        query_embedding=embed_query(question),
        top_k=config.CANDIDATE_K,
        document_filter=document_filter,
    )
    if not config.USE_HYBRID:
        return vector_hits

    keyword_hits = keyword_search.search(
        corpus=vector_store.all_chunks(persist_dir, document_filter),
        query=question,
        top_k=config.CANDIDATE_K,
    )
    return _rrf_fuse([vector_hits, keyword_hits])
