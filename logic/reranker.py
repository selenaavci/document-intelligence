from __future__ import annotations

from functools import lru_cache

from logic import config
from logic.vector_store import RetrievedChunk


@lru_cache(maxsize=2)
def _get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Adayları soruya uygunluğa göre yeniden sıralar. Model tanımlı değilse
    mevcut sırayı kısaltarak döndürür (reranking devre dışı)."""
    if not candidates or not config.RERANK_MODEL:
        return candidates[:top_k]
    model = _get_reranker(config.RERANK_MODEL)
    scores = model.predict([(query, c.text) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    out = []
    for chunk, score in ranked[:top_k]:
        chunk.score = float(score)
        out.append(chunk)
    return out
