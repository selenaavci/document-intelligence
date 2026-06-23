from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from logic.vector_store import RetrievedChunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def search(corpus: list[RetrievedChunk], query: str, top_k: int) -> list[RetrievedChunk]:
    """Korpus içinde BM25 ile anahtar kelime araması yapar."""
    if not corpus:
        return []
    bm25 = BM25Okapi([_tokenize(c.text) for c in corpus])
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
