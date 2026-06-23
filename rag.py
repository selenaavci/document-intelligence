from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from logic import config, llm, reranker, retrieval
from logic.vector_store import RetrievedChunk


SYSTEM_PROMPT = (
    "Sen verilen dokümanlar üzerinden soru cevaplayan bir asistansın. "
    "Cevabını yalnızca sağlanan PASAJ'lara dayandır. "
    "Pasajlarda cevap yoksa açıkça 'Sağlanan dokümanlarda bu bilgi yok.' de. "
    "Cevap verirken kaynak pasajların numarasını köşeli parantezle belirt: örn. [1], [2]. "
    "Türkçe sorulara Türkçe, İngilizce sorulara İngilizce cevap ver."
)

REWRITE_PROMPT = (
    "Aşağıdaki sohbet geçmişine bakarak, son soruyu tek başına anlaşılır "
    "(bağlamdan bağımsız) tek bir soru olacak şekilde yeniden yaz. "
    "Sadece yeniden yazılmış soruyu döndür, başka hiçbir şey yazma."
)


@dataclass
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk]


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (Kaynak: {c.document_name}, sayfa {c.page_number})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )


def _standalone_question(question: str, history: list[dict]) -> str:
    if not (config.USE_QUERY_REWRITE and history):
        return question
    try:
        rewritten = llm.chat(messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            *history,
            {"role": "user", "content": question},
        ])
        return rewritten.strip() or question
    except Exception:
        return question


def answer_question(
    question: str,
    persist_dir: Path,
    history: list[dict] | None = None,
    top_k: int = config.TOP_K,
    document_filter: list[str] | None = None,
    model: str | None = None,
) -> RagAnswer:
    search_query = _standalone_question(question, history or [])
    candidates = retrieval.get_candidates(search_query, persist_dir, document_filter)
    chunks = reranker.rerank(search_query, candidates, top_k)

    if not chunks:
        return RagAnswer(
            answer="Henüz yüklenmiş bir doküman yok veya soruyla eşleşen pasaj bulunamadı.",
            sources=[],
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"PASAJLAR:\n{_format_context(chunks)}\n\nSORU: {question}",
    })

    return RagAnswer(answer=llm.chat(messages=messages, model=model), sources=chunks)
