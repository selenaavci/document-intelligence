from __future__ import annotations

from dataclasses import dataclass

from logic.document_loader import DocumentPage


@dataclass
class Chunk:
    text: str
    page_number: int
    chunk_index: int


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # Cümle/paragraf sınırına çek (varsa)
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(sep, start + max_chars // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in parts if p]


def chunk_pages(
    pages: list[DocumentPage],
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    """Sayfa listesini örtüşmeli, paragraf-aware parçalara böler."""
    chunks: list[Chunk] = []
    idx = 0
    for page in pages:
        for part in _split_long_text(page.text, max_chars, overlap):
            chunks.append(Chunk(text=part, page_number=page.page_number, chunk_index=idx))
            idx += 1
    return chunks
