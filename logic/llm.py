from __future__ import annotations

import json
import ssl
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from logic import config


# AI Hub standardı: kurum içi / lokal dil modeli altyapısına standart bir HTTP
# uç noktası (".../chat/completions") üzerinden bağlanılır. Bağlantı bilgileri
# (adres, model, anahtar) config.py'den gelir — bu dosyaya dokunmana gerek yok.

_TIMEOUT = 90.0


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Dil modelinden sohbet yanıtı alır.

    messages: [{role: 'system'|'user'|'assistant', content: str}]
    """
    base_url = config.LLM_BASE_URL.strip().rstrip("/")
    if not base_url or not (model or config.LLM_MODEL):
        raise RuntimeError(
            "LLM ayarları eksik. logic/config.py içindeki LLM_BASE_URL ve "
            "LLM_MODEL alanlarını doldur."
        )

    payload = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
    }
    headers = {"Content-Type": "application/json"}
    if config.LLM_API_KEY.strip():
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY.strip()}"

    req = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(req, timeout=_TIMEOUT, context=ssl.create_default_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM sunucusu hata döndürdü ({e.code}): {body}") from e
    except URLError as e:
        raise RuntimeError(
            f"LLM sunucusuna bağlanılamadı: {e.reason}. "
            "config.py'deki LLM_BASE_URL adresini kontrol et."
        ) from e

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"LLM yanıtı beklenmedik biçimde: {e} | {data!r}") from e
