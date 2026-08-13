"""Cosmos-Embed1-448p 的用戶端：透過 HTTP 呼叫 cosmos_embed_server.py（獨立 venv 的 sidecar，
見該檔案開頭說明為什麼要分開一個 process）。這裡的 CosmosEmbedder 只實作
app.py/agent_search.py 實際會用到的 .embed_text()，介面跟 ClipEmbedder 一致，讓
app.py 的 get_embedder() 可以透明地依 model_key 換用不同實作。

影格/clip 的 embedding（ingest 用）走 ingest_cosmos_embed.py 直接呼叫 sidecar 的
/embed_clips，不經過這個類別。
"""
import json
import urllib.error
import urllib.request

import config


class CosmosEmbedError(RuntimeError):
    pass


class CosmosEmbedder:
    def __init__(self, url: str = None):
        self.url = (url or config.COSMOS_EMBED_SERVER_URL).rstrip("/")
        try:
            urllib.request.urlopen(self.url + "/health", timeout=5)
        except Exception:
            raise CosmosEmbedError(
                f"連不到 Cosmos-Embed1 服務 ({self.url})。請先啟動：bash serve_cosmos_embed.sh"
            )

    def _post(self, path: str, payload: dict, timeout: int = 60) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise CosmosEmbedError(f"Cosmos-Embed1 服務錯誤 {e.code}：{body[:300]}")

    def embed_text(self, text: str) -> list[float]:
        resp = self._post("/embed_text", {"texts": [text]})
        return resp["embeddings"][0]
