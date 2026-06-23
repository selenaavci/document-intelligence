import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from logic import config
from logic.chunker import chunk_pages
from logic.document_loader import SUPPORTED_EXTENSIONS, load_document
from logic.embeddings import embed_texts
from logic.rag import answer_question
from logic.vector_store import (
    add_document,
    clear_all,
    delete_document,
    list_documents,
)


PERSIST_DIR = config.PERSIST_DIR


st.set_page_config(page_title="Doküman Asistanı", layout="wide")
st.title("Doküman Asistanı")
st.caption(
    "Dokümanlarını yükle, içerikleri hakkında günlük dille soru sor. "
    "Yanıtlar yalnızca senin yüklediğin belgelere dayanır."
)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = {}


def render_sources(sources) -> None:
    """Bir yanıtın hangi doküman bölümlerine dayandığını gösterir."""
    if not sources:
        return
    with st.expander(f"Yanıtın dayandığı doküman bölümleri ({len(sources)})"):
        for c in sources:
            st.markdown(f"**{c.document_name} — sayfa {c.page_number}**")
            st.caption(c.text[:800] + ("…" if len(c.text) > 800 else ""))


with st.sidebar:
    st.header("Doküman yükle")
    uploaded = st.file_uploader(
        "PDF, Word, metin (TXT), Markdown veya web sayfası (HTML) dosyaları",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Dokümanları ekle", type="primary", use_container_width=True):
        progress = st.progress(0.0)
        for i, f in enumerate(uploaded):
            try:
                pages = load_document(f.name, f)
                if not pages:
                    st.warning(f"'{f.name}' boş görünüyor veya içinden metin okunamadı.")
                    continue
                chunks = chunk_pages(
                    pages,
                    max_chars=config.CHUNK_SIZE,
                    overlap=config.CHUNK_OVERLAP,
                )
                vectors = embed_texts([c.text for c in chunks])
                add_document(PERSIST_DIR, f.name, chunks, vectors)
                st.success(f"'{f.name}' eklendi.")
            except Exception as e:
                st.error(f"'{f.name}' eklenirken bir sorun oluştu: {e}")
            progress.progress((i + 1) / len(uploaded))

    st.divider()
    st.header("Yüklenen dokümanlar")
    docs = list_documents(PERSIST_DIR)
    if not docs:
        st.info("Henüz doküman yüklenmedi.")
        selected_docs = None
    else:
        names = [d["name"] for d in docs]
        for d in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"• {d['name']}")
            if col2.button("Sil", key=f"del_{d['name']}"):
                delete_document(PERSIST_DIR, d["name"])
                st.rerun()
        selected_docs = st.multiselect(
            "Yalnızca belirli dokümanlarda ara (boş bırakırsan hepsinde arar)",
            options=names,
            default=[],
        )
        if st.button("Tüm dokümanları temizle", use_container_width=True):
            clear_all(PERSIST_DIR)
            st.session_state.messages = []
            st.session_state.last_sources = {}
            st.rerun()


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(st.session_state.last_sources.get(i))


if prompt := st.chat_input("Dokümanların hakkında bir soru yaz…"):
    if not list_documents(PERSIST_DIR):
        st.warning("Önce sol taraftan en az bir doküman yüklemelisin.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Yanıt hazırlanıyor…"):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                result = answer_question(
                    question=prompt,
                    persist_dir=PERSIST_DIR,
                    history=history,
                    top_k=config.TOP_K,
                    document_filter=selected_docs or None,
                )
            except Exception as e:
                st.error(f"Yanıt alınamadı: {e}")
                st.stop()

        st.markdown(result.answer)
        idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": result.answer})
        st.session_state.last_sources[idx] = result.sources
        render_sources(result.sources)
