"""語音轉文字：單純轉發到 OpenRouter 的 Whisper Large V3 Turbo。

伺服器保管 OPENROUTER_API_KEY，呼叫 /api/transcribe 的人不需要、也看不到這把 key —— 純轉發代理。
"""
import base64
import json
import urllib.request
import urllib.error

import config


class TranscribeError(RuntimeError):
    pass


def transcribe(audio_bytes: bytes, fmt: str) -> dict:
    if not config.OPENROUTER_API_KEY:
        raise TranscribeError("尚未設定 OPENROUTER_API_KEY（見 local_secrets.py 或環境變數）")
    payload = json.dumps({
        "model": config.OPENROUTER_TRANSCRIBE_MODEL,
        "input_audio": {"data": base64.b64encode(audio_bytes).decode(), "format": fmt},
    }).encode()
    req = urllib.request.Request(
        config.OPENROUTER_TRANSCRIBE_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://4070ti.scsonic.com",
            "X-Title": "VSS Whisper Transcribe",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise TranscribeError(f"OpenRouter API 錯誤 {e.code}：{body[:500]}")
