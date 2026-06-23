from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import chromadb

from logic.chunker import Chunk


COLLECTION_NAME = "documents"


@dataclass
class RetrievedChunk:
    text: str
    document_name: str
    page_number: int
    score: float


def _client(persist_dir: Path) -> chromadb.api.ClientAPI:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _collection(persist_dir: Path):
    return _client(persist_dir).get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(document_name: str, chunk: Chunk) -> str:
    h = hashlib.sha1(f"{document_name}::{chunk.chunk_index}::{chunk.text[:64]}".encode()).hexdigest()
    return f"{document_name}::{h[:16]}"


def add_document(
    persist_dir: Path,
    document_name: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> int:
    if not chunks:
        return 0
    coll = _collection(persist_dir)
    ids = [_chunk_id(document_name, c) for c in chunks]
    metadatas = [
        {"document": document_name, "page": c.page_number, "chunk_index": c.chunk_index}
        for c in chunks
    ]
    documents = [c.text for c in chunks]
    coll.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    return len(ids)


def query(
    persist_dir: Path,
    query_embedding: list[float],
    top_k: int = 5,
    document_filter: list[str] | None = None,
) -> list[RetrievedChunk]:
    coll = _collection(persist_dir)
    where = {"document": {"$in": document_filter}} if document_filter else None
    res = coll.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )
    out: list[RetrievedChunk] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for text, meta, dist in zip(docs, metas, dists):
        out.append(RetrievedChunk(
            text=text,
            document_name=str(meta.get("document", "")),
            page_number=int(meta.get("page", 1)),
            score=float(1 - dist),  # cosine distance → similarity
        ))
    return out


def all_chunks(
    persist_dir: Path,
    document_filter: list[str] | None = None,
) -> list[RetrievedChunk]:
    """BM25 anahtar kelime araması için (filtreli) tüm pasajları döndürür."""
    coll = _collection(persist_dir)
    where = {"document": {"$in": document_filter}} if document_filter else None
    res = coll.get(where=where, include=["documents", "metadatas"])
    out: list[RetrievedChunk] = []
    for text, meta in zip(res.get("documents", []), res.get("metadatas", [])):
        out.append(RetrievedChunk(
            text=text,
            document_name=str(meta.get("document", "")),
            page_number=int(meta.get("page", 1)),
            score=0.0,
        ))
    return out


def list_documents(persist_dir: Path) -> list[dict]:
    """Saklanan her doküman için ad ve chunk sayısı."""
    coll = _collection(persist_dir)
    all_items = coll.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in all_items.get("metadatas", []):
        name = str(meta.get("document", ""))
        if name:
            counts[name] = counts.get(name, 0) + 1
    return [{"name": k, "chunks": v} for k, v in sorted(counts.items())]


def delete_document(persist_dir: Path, document_name: str) -> int:
    coll = _collection(persist_dir)
    existing = coll.get(where={"document": document_name})
    ids = existing.get("ids", [])
    if ids:
        coll.delete(ids=ids)
    return len(ids)


def clear_all(persist_dir: Path) -> None:
    client = _client(persist_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
