from __future__ import annotations

import re
from dataclasses import dataclass

from logic.document_loader import DocumentPage


@dataclass
class Chunk:
    text: str
    page_number: int
    chunk_index: int


# Mevzuat madde başlıkları: "MADDE 5", "Madde 5", "GEÇİCİ MADDE 1", "EK MADDE 2".
# Bir madde başlığı yeni bir mantıksal birim demektir; mümkünse buradan bölünür.
_ARTICLE_RE = re.compile(
    r"(?=^\s*(?:GEÇİCİ\s+|EK\s+)?MADDE\s+\d+)",
    re.IGNORECASE | re.MULTILINE,
)

# Cümle sonu: nokta/soru/ünlem (+ opsiyonel kapanış tırnağı) ardından boşluk ve
# büyük harf/rakam ile başlayan yeni cümle. Ondalık sayı (1.200) ve satır içi
# kısaltmalar genelde ardından boşluk+büyük harf gelmediği için bölünmez.
_SENTENCE_RE = re.compile(
    r"""(?<=[.!?…])["'”’\)\]]*\s+(?=[A-ZÇĞİÖŞÜ0-9])""",
    re.VERBOSE,
)


def _split_sentences(text: str) -> list[str]:
    """Metni, asla ortadan bölmeyeceğimiz atomik cümlelere ayırır.

    Satır sonları yapısal sınır kabul edilir (mevzuatta fıkra/bent ayracı),
    ardından her satır cümlelere bölünür. Boş parçalar atılır.
    """
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for sent in _SENTENCE_RE.split(line):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Tek başına max_chars'tan uzun (nadir) bir cümleyi kelime sınırından böler."""
    parts: list[str] = []
    start = 0
    while start < len(sentence):
        end = min(len(sentence), start + max_chars)
        if end < len(sentence):
            cut = sentence.rfind(" ", start + max_chars // 2, end)
            if cut != -1:
                end = cut
        parts.append(sentence[start:end].strip())
        start = end
    return [p for p in parts if p]


def _pack(sentences: list[str], max_chars: int, overlap: int) -> list[str]:
    """Cümleleri max_chars'ı aşmayan, cümle-bütünlüğü korunan parçalara yerleştirir.

    Örtüşme cümle bazlıdır: yeni parça, önceki parçanın son cümle(ler)iyle başlar,
    böylece hiçbir parça cümle ortasından başlamaz/bitmez.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        # Tek başına çok uzun cümle: önce mevcut parçayı kapat, sonra sert böl.
        if len(sent) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(_hard_split(sent, max_chars))
            continue

        addition = len(sent) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            chunks.append(" ".join(current))
            # Cümle bazlı örtüşme: sondan başlayarak overlap kadar cümle taşı.
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) > overlap and carry:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            current = carry
            current_len = sum(len(s) + 1 for s in current)

        current.append(sent)
        current_len += addition

    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if c.strip()]


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Bir sayfa metnini madde-farkındalıklı, cümle-bütünlüklü parçalara böler."""
    if len(text) <= max_chars:
        stripped = text.strip()
        return [stripped] if stripped else []

    # Önce madde sınırlarından bloklara ayır; her blok kendi içinde paketlenir.
    blocks = [b for b in _ARTICLE_RE.split(text) if b.strip()]
    if not blocks:
        blocks = [text]

    parts: list[str] = []
    for block in blocks:
        sentences = _split_sentences(block)
        if not sentences:
            continue
        parts.extend(_pack(sentences, max_chars, overlap))
    return parts


def chunk_pages(
    pages: list[DocumentPage],
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    """Sayfa listesini madde-farkındalıklı, cümle-bütünlüğü korunan parçalara böler."""
    chunks: list[Chunk] = []
    idx = 0
    for page in pages:
        for part in _split_long_text(page.text, max_chars, overlap):
            chunks.append(Chunk(text=part, page_number=page.page_number, chunk_index=idx))
            idx += 1
    return chunks
