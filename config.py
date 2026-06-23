"""
Doküman Asistanı — tüm ayarlar burada. Arayüzde teknik ayar yoktur.
Şirket bilgisayarında en çok "1) LLM AYARLARI" bölümünü değiştirirsin.
"""

from pathlib import Path


# 1) LLM AYARLARI ----------------------------------------------------------
# Kurum içi / lokal dil modeline ".../chat/completions" HTTP uç noktasından
# bağlanılır. Bu üç bilgiyi BT / sistem ekibinden alabilirsin.

LLM_BASE_URL = ""   # örn. "http://10.0.0.5:8000/v1"
LLM_MODEL = ""      # altyapında tanımlı modelin adı
LLM_API_KEY = ""    # gerekmiyorsa boş bırak
LLM_TEMPERATURE = 0.2


# 2) EMBEDDING MODELİ ------------------------------------------------------
# Metni anlamca aranabilir vektörlere çevirir (TR + EN). İlk çalıştırmada iner.
EMBEDDING_MODEL = ""   # örn. "intfloat/multilingual-e5-small"


# 3) RERANKER MODELİ -------------------------------------------------------
# Aday pasajları soruya göre yeniden sıralar (kalitenin en çok arttığı adım).
# Boş bırakırsan reranking devre dışı kalır, sadece arama skorları kullanılır.
RERANK_MODEL = ""      # örn. "BAAI/bge-reranker-v2-m3"


# 4) VERİ KLASÖRÜ ----------------------------------------------------------
PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"


# 5) ARAMA / RANKING AYARLARI ---------------------------------------------
TOP_K = 5              # LLM'e verilecek nihai pasaj sayısı
CANDIDATE_K = 30       # rerank öncesi toplanan aday sayısı (vektör + anahtar kelime)
USE_HYBRID = True      # vektör + BM25 anahtar kelime aramasını birleştir
USE_QUERY_REWRITE = True  # takip sorularını geçmişe göre bağımsız hale getir

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
