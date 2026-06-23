from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from logic import config


@lru_cache(maxsize=4)
def _get_embedder(model_name: str) -> SentenceTransformer:
    if not model_name:
        raise RuntimeError("EMBEDDING_MODEL boş. logic/config.py içinde bir model adı gir.")
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Belgeleri vektörlere çevirir. e5 modeli 'passage:' önekini bekler."""
    model = _get_embedder(config.EMBEDDING_MODEL)
    vectors = model.encode(
        [f"passage: {t}" for t in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    model = _get_embedder(config.EMBEDDING_MODEL)
    vector = model.encode(
        [f"query: {text}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector[0].tolist()
